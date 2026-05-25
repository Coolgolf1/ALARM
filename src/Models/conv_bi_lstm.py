import torch
import torch.nn as nn


class ConvBiLSTM(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conv_channels: int,
        lstm_hidden: int,
        num_classes: int,
        kernel_size: int = 5,
        pool_size: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embedding_dim: int = embedding_dim

        self.conv = nn.Conv1d(
            in_channels=self.embedding_dim,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size)

        self.batch_norm = nn.BatchNorm1d(conv_channels)

        self.dropout1 = nn.Dropout(dropout)

        self.bilstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout2 = nn.Dropout(dropout)

        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (b, seq, embed_dim)

        # (b, embed_dim, seq)
        embedded_inputs = inputs.transpose(1, 2)

        # (b, conv_c, conv_seq)
        conv: torch.Tensor = self.conv(embedded_inputs)
        activated: torch.Tensor = self.relu(conv)

        # (b, conv_c, pool_seq)
        pooled: torch.Tensor = self.pool(activated)
        normalised: torch.Tensor = self.batch_norm(pooled)
        dropout1: torch.Tensor = self.dropout1(normalised)

        # (b, pool_seq, conv_c)
        bilstm_input: torch.Tensor = dropout1.transpose(1, 2)

        # (b, pool_seq, 2 * hidden_dim)
        # bilstm_out: torch.Tensor
        bilstm_out, _ = self.bilstm(bilstm_input)

        # (b, 2 * hidden_dim)
        bilstm_out_mean: torch.Tensor = bilstm_out.mean(dim=1)

        # (b, 2 * hidden_dim)
        dropout2: torch.Tensor = self.dropout2(bilstm_out_mean)

        # (b, num_classes)
        output: torch.Tensor = self.fc(dropout2)

        return output
