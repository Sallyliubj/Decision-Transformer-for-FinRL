import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DecisionTransformerConfig, DecisionTransformerModel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config_tickers import DOW_30_TICKER
from finrl.config import INDICATORS
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

# Global constants
CONTEXT_LENGTH = 20
TRAIN_EPOCHS = 1000
TRAIN_LR = 3e-5
MODEL_PATH = f"decision_transformer_{CONTEXT_LENGTH}_lr_{TRAIN_LR}_epochs_{TRAIN_EPOCHS}.pth"
TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2018-12-31'
TRADE_START_DATE = '2019-01-01'
TRADE_END_DATE = '2020-12-31'
BENCHMARK_TICKER = 'QQQ' # Nasdaq 100


def parse_array(s):
    """
    Parses a string representation of a numpy array by direct manipulation.
    Handles multiple spaces, newlines, and scientific notation.
    """
    try:
        s_cleaned = s.replace('[', '').replace(']', '').replace('\n', '').strip()
        parts = s_cleaned.split(' ')
        return np.array([float(p) for p in parts if p != ''])
    except Exception as e:
        print(f"FATAL ERROR: Could not parse string: '{s}'")
        raise e

class FinRLDataset(Dataset):
    def __init__(self, df: pd.DataFrame, context_len: int):
        self.df = df
        self.context_len = context_len
        self.df['episode_start'] = self.df['episode_start'].astype(bool)
        self.episode_starts = self.df.index[self.df['episode_start']].tolist()
        self.num_episodes = len(self.episode_starts)
        self.state_dim = len(df['state'].iloc[0])
        self.action_dim = len(df['action'].iloc[0])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        episode_idx = np.searchsorted(self.episode_starts, idx, side='right') - 1
        episode_start_idx = self.episode_starts[episode_idx]
        start_seq = idx
        end_seq = start_seq + self.context_len
        seq_df = self.df.iloc[start_seq:end_seq]
        
        states = torch.from_numpy(np.vstack(seq_df['state'].values)).float()
        actions = torch.from_numpy(np.vstack(seq_df['action'].values)).float()
        rewards = torch.from_numpy(seq_df['reward'].values).float().view(-1, 1)
        returns_to_go = torch.from_numpy(seq_df['return_to_go'].values).float().view(-1, 1)
        timesteps = torch.arange(start_seq - episode_start_idx, (start_seq - episode_start_idx) + len(seq_df)).long()
        
        seq_len = len(seq_df)
        padding_len = self.context_len - seq_len
        attention_mask = torch.cat([torch.ones(seq_len), torch.zeros(padding_len)]).long()

        if padding_len > 0:
            states = torch.cat([states, torch.zeros(padding_len, self.state_dim)], dim=0)
            actions = torch.cat([actions, torch.zeros(padding_len, self.action_dim)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(padding_len, 1)], dim=0)
            returns_to_go = torch.cat([returns_to_go, torch.zeros(padding_len, 1)], dim=0)
            timesteps = torch.cat([timesteps, torch.zeros(padding_len, dtype=torch.long)], dim=0)
            
        return {
            "states": states, "actions": actions, "rewards": rewards,
            "returns_to_go": returns_to_go, "timesteps": timesteps, "attention_mask": attention_mask
        }

def train(epochs = 100, lr = 1e-2):
    print("--- Starting Training ---")
    df = pd.read_csv("decision_transformer_ready_dataset.csv")
    df['state'] = df['state'].apply(parse_array)
    df['action'] = df['action'].apply(parse_array)

    df['episode_start'] = df['episode_start'].astype(bool)
    episode_starts = df.index[df['episode_start']].tolist()
    episode_indices = list(range(len(episode_starts)))
    train_ep_indices, val_ep_indices = train_test_split(episode_indices, test_size=0.2, random_state=42)

    train_df_indices = []
    for ep_idx in train_ep_indices:
        start = episode_starts[ep_idx]
        end = episode_starts[ep_idx+1] if ep_idx + 1 < len(episode_starts) else len(df)
        train_df_indices.extend(range(start, end))

    train_df = df.iloc[train_df_indices].reset_index(drop=True)

    val_df_indices = []
    for ep_idx in val_ep_indices:
        start = episode_starts[ep_idx]
        end = episode_starts[ep_idx+1] if ep_idx + 1 < len(episode_starts) else len(df)
        val_df_indices.extend(range(start, end))
    val_df = df.iloc[val_df_indices].reset_index(drop=True)

    train_dataset = FinRLDataset(train_df, context_len=CONTEXT_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_dataset = FinRLDataset(val_df, context_len=CONTEXT_LENGTH)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    state_dim = train_dataset.state_dim
    act_dim = train_dataset.action_dim

    config = DecisionTransformerConfig(state_dim=state_dim, act_dim=act_dim, hidden_size=128, n_layer=3, n_head=1, n_inner=4*128)
    model = DecisionTransformerModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    from torch.optim.lr_scheduler import StepLR
    loss_fn = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    CLIP_GRAD_NORM = 1.0

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            states, actions, returns_to_go, timesteps, attention_mask = (
                batch["states"].to(device), batch["actions"].to(device),
                batch["returns_to_go"].to(device), batch["timesteps"].to(device),
                batch["attention_mask"].to(device)
            )
            
            optimizer.zero_grad()
            
            action_preds = model(
                states=states, 
                actions=actions, 
                returns_to_go=returns_to_go, 
                timesteps=timesteps, 
                attention_mask=attention_mask
            ).action_preds
            
            action_target = actions[attention_mask > 0]
            action_pred = action_preds[attention_mask > 0]
            
            loss = loss_fn(action_pred, action_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            optimizer.step()    
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                states, actions, returns_to_go, timesteps, attention_mask = (
                    batch["states"].to(device), batch["actions"].to(device),
                    batch["returns_to_go"].to(device), batch["timesteps"].to(device),
                    batch["attention_mask"].to(device)
                )
                
                action_preds = model(
                    states=states, 
                    actions=actions, 
                    returns_to_go=returns_to_go, 
                    timesteps=timesteps, 
                    attention_mask=attention_mask
                ).action_preds
                
                action_target = actions[attention_mask > 0]
                action_pred = action_preds[attention_mask > 0]
                
                val_loss = loss_fn(action_pred, action_target)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{epochs}, Avg Train Loss: {avg_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"New best model saved to {MODEL_PATH} with validation loss: {best_val_loss:.4f}")

    print("--- Training Finished ---")
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_plot.png')
    print("Loss plot saved to training_loss_plot.png")

def evaluate():
    print("\n--- Starting Evaluation ---")
    df = pd.read_csv("decision_transformer_ready_dataset.csv")
    df['state'] = df['state'].apply(parse_array)
    df['action'] = df['action'].apply(parse_array)

    state_dim = len(df['state'].iloc[0])
    act_dim = len(df['action'].iloc[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DecisionTransformerConfig(state_dim=state_dim, act_dim=act_dim, hidden_size=128, n_layer=3, n_head=1, n_inner=4*128)
    model = DecisionTransformerModel(config)
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(device)
    
    trade_df = pd.read_csv('trade_data.csv')
    trade_df = trade_df.set_index(trade_df.columns[0])
    trade_df.index.names = ['']

    stock_dimension = len(trade_df.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    eval_env = StockTradingEnv(df=trade_df, **env_kwargs)

    def evaluate_episode_rtg(model, env, target_return):
        model.eval()
        state, _ = env.reset()
        state = np.array(state)
        done = False
        
        states = torch.from_numpy(state).reshape(1, 1, state_dim).float().to(device)
        actions = torch.zeros((1, 1, act_dim), device=device)
        rewards_history = torch.zeros(1, 1, device=device)
        target_return_tensor = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, 1, 1)
        timesteps = torch.tensor(0, device=device).reshape(1, 1)
        episode_return, episode_length = 0, 0
        
        while not done:
            seq_len = states.shape[1]
            attention_mask = torch.ones(1, seq_len, device=device, dtype=torch.long)
            
            action_preds = model(
                states=states,
                actions=actions,
                rewards=rewards_history,
                returns_to_go=target_return_tensor,
                timesteps=timesteps,
                attention_mask=attention_mask
            ).action_preds
            
            action = action_preds[0, -1].detach().cpu().numpy()
            state, reward, terminated, truncated, _ = env.step(action)
            state = np.array(state)
            done = terminated or truncated

            actions = torch.cat([actions, torch.from_numpy(action).reshape(1, 1, act_dim).float().to(device)], dim=1)
            states = torch.cat([states, torch.from_numpy(state).reshape(1, 1, state_dim).float().to(device)], dim=1)
            rewards_history = torch.cat([rewards_history, torch.tensor([[reward]], device=device, dtype=torch.float32)], dim=1)
            
            reward_tensor = torch.tensor([[[reward]]], device=device, dtype=torch.float32)
            new_rtg = target_return_tensor[:, -1:, :] - reward_tensor
            target_return_tensor = torch.cat([target_return_tensor, new_rtg], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (episode_length + 1)], dim=1)
            episode_return += reward
            episode_length += 1

            states, actions, rewards_history, target_return_tensor, timesteps = (
                states[:, -CONTEXT_LENGTH:],
                actions[:, -CONTEXT_LENGTH:],
                rewards_history[:, -CONTEXT_LENGTH:],
                target_return_tensor[:, -CONTEXT_LENGTH:],
                timesteps[:, -CONTEXT_LENGTH:]
            )
            
        return env.save_asset_memory()

    df['episode_start'] = df['episode_start'].astype(bool)
    episode_starts = df.index[df['episode_start']].tolist()
    episode_ends = episode_starts[1:] + [len(df)]
    episode_returns = [df.iloc[start:end]['reward'].sum() for start, end in zip(episode_starts, episode_ends)]
    target_return_prompt = np.mean(episode_returns)

    print(f"\nEvaluating with a Target Return of: {target_return_prompt:.2f}")
    
    account_value_df = evaluate_episode_rtg(model, eval_env, target_return_prompt)

    start_date = account_value_df.date[0]
    end_date = account_value_df.date[len(account_value_df)-1]
    
    baseline_df = get_baseline(
        ticker=BENCHMARK_TICKER,
        start=start_date,
        end=end_date
    )

    baseline_df['baseline'] = baseline_df['close'] / baseline_df['close'].iloc[0] * account_value_df['account_value'].iloc[0]
    account_value_with_baseline = pd.merge(account_value_df, baseline_df[['baseline']], how='left', left_index=True, right_index=True)

    account_value_to_plot = account_value_with_baseline.rename(columns={'index': 'date'})

    print("Calculating performance stats...")
    perf_stats_all = backtest_stats(account_value=account_value_to_plot)
    print("--- Performance Stats ---")
    print(perf_stats_all)

    annual_return = f"{perf_stats_all.loc['Annual return']*100:.2f}%"
    sharpe_ratio = f"{perf_stats_all.loc['Sharpe ratio']:.2f}"
    max_drawdown = f"{perf_stats_all.loc['Max drawdown']*100:.2f}%"
    volatility = f"{perf_stats_all.loc['Annual volatility']*100:.2f}%"
    total_return = ( (account_value_to_plot['account_value'].iloc[-1] / account_value_to_plot['account_value'].iloc[0]) - 1 ) * 100
    total_return_str = f"{total_return:.2f}%"

    stats_text = (
        f"Total Return: {total_return_str}\n"
        f"Annual Return: {annual_return}\n"
        f"Sharpe Ratio: {sharpe_ratio}\n"
        f"Max Drawdown: {max_drawdown}\n"
        f"Volatility: {volatility}"
    )
    print(stats_text)

    print("Generating backtest plot...")
    fig, ax = plt.subplots(figsize=(20, 8))
    account_value_to_plot['date'] = pd.to_datetime(account_value_to_plot['date'])

    ax.plot(account_value_to_plot['date'], account_value_to_plot['account_value'], lw=2, label='Decision Transformer')
    ax.plot(account_value_to_plot['date'], account_value_to_plot['baseline'], lw=2, linestyle='--', label=f'{BENCHMARK_TICKER} Benchmark')

    ax.set_title('Portfolio Value vs. Benchmark', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.yaxis.set_major_formatter('${x:,.0f}')
    
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    ax.set_xticks([start_date_dt, end_date_dt])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))

    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)

    final_value = account_value_to_plot['account_value'].iloc[-1]
    ax.text(account_value_to_plot['date'].iloc[-1], final_value, f' ${final_value:,.2f}',
            verticalalignment='center', fontsize=12, fontweight='bold', ha='right')
            
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

    plt.tight_layout()
    fig.savefig('backtest_plot.png', dpi=300)
    print("Plot saved to backtest_plot.png")


if __name__ == '__main__':
    train(epochs = TRAIN_EPOCHS, lr = TRAIN_LR)
    evaluate() 