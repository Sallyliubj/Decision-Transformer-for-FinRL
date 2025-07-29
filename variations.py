import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DecisionTransformerConfig, DecisionTransformerModel
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config_tickers import DOW_30_TICKER
from finrl.config import INDICATORS
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

# Global constants
CONTEXT_LENGTH = 20
NUM_EPOCHS = 500
LR = 1e-3

TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2018-12-31'
TRADE_START_DATE = '2019-01-01'
TRADE_END_DATE = '2020-12-31'

BENCHMARK_TICKER_QQQ = 'QQQ' # Nasdaq 100
BENCHMARK_TICKER_DOW = 'DIA' # Dow Jones Industrial Average



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
        # Ensure 'state' and 'action' columns exist before accessing them
        if 'state' not in df.columns or 'action' not in df.columns:
            raise ValueError("DataFrame must contain 'state' and 'action' columns.")
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

class NoRtgDT(nn.Module):
    """
    A Decision Transformer model that operates WITHOUT the return-to-go signal.
    It still uses the core attention mechanism. This is the correct model
    for the 'no_rtg' model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.embed_timestep = nn.Embedding(config.max_ep_len, self.hidden_size)
        self.embed_state = nn.Linear(config.state_dim, self.hidden_size)
        self.embed_action = nn.Linear(config.act_dim, self.hidden_size)
        self.embed_ln = nn.LayerNorm(self.hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=config.n_head,
            dim_feedforward=4 * self.hidden_size, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layer)

        self.predict_action = nn.Sequential(
            nn.Linear(self.hidden_size, config.act_dim),
            nn.Tanh()
        )

    def forward(self, states, actions, timesteps, attention_mask, **kwargs):
        batch_size, seq_length = states.shape[0], states.shape[1]

        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings

        # We interleave state and action embeddings
        # (s_1, a_1, s_2, a_2, ...)
        stacked_inputs = torch.stack((state_embeddings, action_embeddings), dim=1).permute(0, 2, 1, 3).reshape(batch_size, 2 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Create a causal mask for the transformer
        # We need to reshape the attention mask to match the interleaved sequence length
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 2 * seq_length)
        )
        padding_mask = (stacked_attention_mask == 0)
        # need a square causal mask to prevent attention to future tokens
        seq_len_2 = 2 * seq_length
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len_2, dtype=torch.bool, device=states.device
        )

        encoder_outputs = self.encoder(
            src=stacked_inputs,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        # We predict the action from the state embedding's output position
        action_preds = self.predict_action(encoder_outputs[:, ::2])
        return {'action_preds': action_preds}
    
    
class Lstm(nn.Module):
    """
    An LSTM model that uses state, action, AND return-to-go as inputs.
    This provides a direct comparison of Attention vs. Recurrence.
    """
    def __init__(self, state_dim, act_dim, hidden_size=128):
        super().__init__()
        # The input size now includes state, action, reward (1D), and RTG (1D)
        self.input_size = state_dim + act_dim + 1 + 1
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, act_dim)

    def forward(self, states, actions, rewards, returns_to_go, **kwargs):
        # Concatenate all four inputs along the last dimension
        x = torch.cat([states, actions, rewards, returns_to_go], dim=-1)
        
        lstm_out, _ = self.lstm(x)
        action_preds = self.fc(lstm_out)
        return {'action_preds': action_preds}


def train(epochs=100, lr=1e-2, model=None):
    print(f"--- Starting Training for Model: {model} ---")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = DecisionTransformerConfig(state_dim=state_dim, act_dim=act_dim, hidden_size=128, n_layer=3, n_head=1, n_inner=4*128, max_ep_len=4096)
    if model == "lstm":
        model = Lstm(state_dim=state_dim, act_dim=act_dim).to(device)
    elif model == "no_rtg":
        model = NoRtgDT(config).to(device)
    else: # Baseline
        model = DecisionTransformerModel(config).to(device)
    
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    CLIP_GRAD_NORM = 1.0

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            states, actions, rewards, returns_to_go, timesteps, attention_mask = (
                batch["states"].to(device), batch["actions"].to(device),
                batch["rewards"].to(device), batch["returns_to_go"].to(device),
                batch["timesteps"].to(device), batch["attention_mask"].to(device)
            )
            
            model_args = {
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "returns_to_go": returns_to_go,
                "timesteps": timesteps,
                "attention_mask": attention_mask,
            }
            
            optimizer.zero_grad()
            outputs = model(**model_args)
            
            action_preds = outputs['action_preds'] if isinstance(outputs, dict) else outputs.action_preds

            action_target = actions[attention_mask > 0]
            action_pred = action_preds[attention_mask > 0]

            loss = loss_fn(action_pred, action_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                states, actions, rewards, returns_to_go, timesteps, attention_mask = (
                    batch["states"].to(device), batch["actions"].to(device),
                    batch["rewards"].to(device), batch["returns_to_go"].to(device),
                    batch["timesteps"].to(device), batch["attention_mask"].to(device)
                )

                model_args = {
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "returns_to_go": returns_to_go,
                "timesteps": timesteps,
                "attention_mask": attention_mask,
            }

                outputs = model(**model_args)
                action_preds = outputs['action_preds'] if isinstance(outputs, dict) else outputs.action_preds

                action_target = actions[attention_mask > 0]
                action_pred = action_preds[attention_mask > 0]

                val_loss = loss_fn(action_pred, action_target)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        # scheduler.step(avg_val_loss)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{epochs}, Avg Train Loss: {avg_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")

        model_path = f"decision_transformer_{model or 'baseline'}.pth"
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved to {model_path} with validation loss: {best_val_loss:.4f}")

    print("--- Training Finished ---")

    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Training & Validation Loss (Model: {model or "Baseline"})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'training_loss_plot_{model or "baseline"}.png')
    print(f"Loss plot saved to training_loss_plot_{model or 'baseline'}.png")

def run_evaluation_for_model(model=None):
    """
    Runs the evaluation for a single model specified by the model type.
    Returns a dataframe with the account values.
    """
    print(f"\n--- Running Evaluation for Model: {model or 'Baseline'} ---")
    
    # Load the offline dataset to determine dimensions and target return
    df = pd.read_csv("decision_transformer_ready_dataset.csv")
    df['state'] = df['state'].apply(parse_array)
    df['action'] = df['action'].apply(parse_array)

    state_dim = len(df['state'].iloc[0])
    act_dim = len(df['action'].iloc[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the appropriate model architecture
    model_path = f"decision_transformer_{model or 'baseline'}.pth"
    
    config = DecisionTransformerConfig(
        state_dim=state_dim, act_dim=act_dim, hidden_size=128,
        n_layer=3, n_head=1, n_inner=4*128, max_ep_len=4096
    )

    # Select the correct model architecture
    if model == "lstm":
        model = Lstm(state_dim=state_dim, act_dim=act_dim)
    elif model == "no_rtg":
        model = NoRtgDT(config)
    else:  # Baseline
        model = DecisionTransformerModel(config)
    
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Set up the trading environment
    trade_df = pd.read_csv('trade_data.csv')
    trade_df = trade_df.set_index(trade_df.columns[0])
    trade_df.index.names = ['']

    stock_dimension = len(trade_df.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    
    env_kwargs = {
        "hmax": 100, "initial_amount": 1000000,
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [0.001] * stock_dimension,
        "sell_cost_pct": [0.001] * stock_dimension,
        "state_space": state_space, "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS, "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    eval_env = StockTradingEnv(df=trade_df, **env_kwargs)

    # Determine the target return from the training data
    df['episode_start'] = df['episode_start'].astype(bool)
    episode_starts = df.index[df['episode_start']].tolist()
    episode_ends = episode_starts[1:] + [len(df)]
    episode_returns = [df.iloc[start:end]['reward'].sum() for start, end in zip(episode_starts, episode_ends)]
    target_return_prompt = np.mean(episode_returns)
    print(f"Evaluating with a Target Return of: {target_return_prompt:.2f}")

    # --- Evaluation Loop ---
    state, _ = eval_env.reset()
    state = np.array(state)
    done = False

    states = torch.from_numpy(state).reshape(1, 1, state_dim).float().to(device)
    actions = torch.zeros((1, 1, act_dim), device=device)
    rewards_history = torch.zeros((1, 1, 1), device=device)
    target_return_tensor = torch.tensor(target_return_prompt, device=device, dtype=torch.float32).reshape(1, 1, 1)
    timesteps = torch.tensor(0, device=device).reshape(1, 1)

    while not done:
        with torch.no_grad():
            # Build the argument dictionary for the model forward pass
            model_args = {
                "states": states, "actions": actions, "rewards": rewards_history,
                "returns_to_go": target_return_tensor, "timesteps": timesteps,
                "attention_mask": torch.ones(1, states.shape[1], device=device)
            }

            outputs = model(**model_args)
            action_preds = outputs['action_preds'] if isinstance(outputs, dict) else outputs.action_preds
            action = action_preds[0, -1].detach().cpu().numpy()

        
        state, reward, terminated, truncated, _ = eval_env.step(action)
        state = np.array(state)
        done = terminated or truncated

        # Update sequences for the next step
        actions = torch.cat([actions, torch.from_numpy(action).reshape(1, 1, act_dim).float().to(device)], dim=1)
        states = torch.cat([states, torch.from_numpy(state).reshape(1, 1, state_dim).float().to(device)], dim=1)
        rewards_history = torch.cat([rewards_history, torch.tensor([[[reward]]], device=device).float()], dim=1)
        
        new_rtg = target_return_tensor[:, -1:, :] - torch.tensor([[[reward]]], device=device).float()
        target_return_tensor = torch.cat([target_return_tensor, new_rtg], dim=1)
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (timesteps.max() + 1)], dim=1)

        # Trim sequences to context length
        states, actions, rewards_history, target_return_tensor, timesteps = (
            states[:, -CONTEXT_LENGTH:], actions[:, -CONTEXT_LENGTH:], rewards_history[:, -CONTEXT_LENGTH:],
            target_return_tensor[:, -CONTEXT_LENGTH:], timesteps[:, -CONTEXT_LENGTH:]
        )
    
    # Return the performance dataframe
    return eval_env.save_asset_memory()

def evaluate_and_plot_all():
    """
    Evaluates all models and plots their performance against benchmarks on a single graph.
    """
    # --- Run evaluations for all models ---
    df_baseline = run_evaluation_for_model(model=None)
    df_lstm = run_evaluation_for_model(model="lstm")
    df_no_rtg = run_evaluation_for_model(model="no_rtg")

    # --- Combine results for plotting ---
    plot_df = pd.DataFrame()
    plot_df['date'] = pd.to_datetime(df_baseline['date'])
    plot_df['Baseline DT'] = df_baseline['account_value'].values
    plot_df['LSTM'] = df_lstm['account_value'].values
    plot_df['No RTG DT'] = df_no_rtg['account_value'].values
    
    # --- Get Benchmark Data ---
    start_date = plot_df['date'].iloc[0]
    end_date = plot_df['date'].iloc[-1]
    initial_value = plot_df['Baseline DT'].iloc[0]

    print("\n--- Fetching Benchmark Data ---")
    qqq_benchmark = get_baseline(ticker='QQQ', start=start_date, end=end_date)
    dow_benchmark = get_baseline(ticker='DIA', start=start_date, end=end_date) # DIA tracks the Dow Jones

    # Normalize benchmarks to start at the same initial value as the portfolios
    plot_df['QQQ'] = qqq_benchmark['close'] / qqq_benchmark['close'].iloc[0] * initial_value
    plot_df['DOW'] = dow_benchmark['close'] / dow_benchmark['close'].iloc[0] * initial_value
    
    print("\n--- Calculating Performance Stats for Plot ---")
    baseline_stats_df = backtest_stats(account_value=df_baseline)
    cumulative_return = f"{baseline_stats_df.loc['Cumulative returns']*100:.2f}%"
    annual_return = f"{baseline_stats_df.loc['Annual return']*100:.2f}%"
    sharpe_ratio = f"{baseline_stats_df.loc['Sharpe ratio']:.2f}"
    max_drawdown = f"{baseline_stats_df.loc['Max drawdown']*100:.2f}%"
    annual_volatility = f"{baseline_stats_df.loc['Annual volatility']*100:.2f}%"
    
    stats_text = (
        f"Baseline DT Performance:\n"
        f"Cumulative Return: {cumulative_return}\n"
        f"Annual Return: {annual_return}\n"
        f"Sharpe Ratio: {sharpe_ratio}\n"
        f"Max Drawdown: {max_drawdown}\n"
        f"Annual Volatility: {annual_volatility}"
    )
    
    # --- Generate the plot ---
    print("\n--- Generating Combined Backtest Plot ---")
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Plot each series
    ax.plot(plot_df['date'], plot_df['Baseline DT'], lw=2, label='Baseline Decision Transformer')
    ax.plot(plot_df['date'], plot_df['LSTM'], lw=2, label='LSTM')
    ax.plot(plot_df['date'], plot_df['No RTG DT'], lw=2, label='No Return-to-Go DT')
    ax.plot(plot_df['date'], plot_df['QQQ'], lw=2, linestyle='--', color='purple', label='Nasdaq 100 (QQQ) Benchmark')
    ax.plot(plot_df['date'], plot_df['DOW'], lw=2, linestyle='--', color='orange', label='Dow Jones (DIA) Benchmark')

    # Formatting the plot
    ax.set_title('Comparative Analysis of Decision Transformer Models vs. Benchmarks', fontsize=18, fontweight='bold')
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Portfolio Value ($)', fontsize=14)
    ax.yaxis.set_major_formatter('${x:,.0f}')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()

    ax.grid(True, which='major', linestyle='--', linewidth=0.6)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    legend_elements = [
        plt.Line2D([0], [0], color='C0', lw=2.5, label=f"Baseline DT: ${plot_df['Baseline DT'].iloc[-1]:,.2f}"),
        plt.Line2D([0], [0], color='C1', lw=2, label=f"LSTM: ${plot_df['LSTM'].iloc[-1]:,.2f}"),
        plt.Line2D([0], [0], color='C2', lw=2, label=f"No RTG DT: ${plot_df['No RTG DT'].iloc[-1]:,.2f}"),
        plt.Line2D([0], [0], color='purple', lw=2, linestyle='--', label=f"QQQ Benchmark"),
        plt.Line2D([0], [0], color='orange', lw=2, linestyle='--', label=f"DOW Benchmark")
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.05, 0.75), fontsize=12)

    plt.tight_layout()
    fig.savefig('backtest_plot_combined.png', dpi=300)
    print("Combined plot saved to backtest_plot_combined.png")
    
    # --- Print final performance stats for each model ---
    print("\n--- Final Performance Stats ---")
    for name, data in [("Baseline", df_baseline), ("LSTM", df_lstm), ("No RTG", df_no_rtg)]:
        print(f"\n--- {name} DT ---")
        stats = backtest_stats(account_value=data)
        print(stats)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=['baseline', 'lstm', 'no_rtg', 'all'],
                        help='Specify which model to train or "all" to train all and plot.')
    args = parser.parse_args()

    if args.model == 'all':
        
        print("--- Training All Models ---")
        train(epochs=NUM_EPOCHS, lr=LR, model=None)
        train(epochs=NUM_EPOCHS, lr=LR, model="lstm")
        train(epochs=NUM_EPOCHS, lr=LR, model="no_rtg")
        
        evaluate_and_plot_all()
        
    elif args.model is not None:
        # Train and evaluate a single specified model
        model_type = None if args.model == 'baseline' else args.model
        train(epochs=NUM_EPOCHS, lr=LR, model=model_type)
        run_evaluation_for_model(model=model_type)
    
    else:
        print("Please specify an model to run or use '--model all'")
    
    
    # example command: python variations.py --model no_rtg