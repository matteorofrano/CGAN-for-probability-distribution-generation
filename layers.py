import torch.nn as nn
import torch



class LayerNormLSTM(nn.Module):
    def __init__(self, input_size:int = 10, hidden_size:int = 20):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_layer = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        self.layer_norm = nn.LayerNorm(4 * hidden_size)

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=-1)
        gates = self.layer_norm(self.input_layer(combined))

        # compute input, forget, cell, output gates
        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c
    




class MultiLayerNormLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size, dropout=0.0):

        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # build one cell per layer
        self.cells = nn.ModuleList()
        for layer in range(n_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.cells.append(LayerNormLSTM(layer_input_size, hidden_size))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # initialize h and c to zeros for every layer
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.n_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.n_layers)]

        for t in range(seq_len):
            # input to the first layer is the actual data at timestep t
            layer_input = x[:, t, :]

            for layer in range(self.n_layers):
                h[layer], c[layer] = self.cells[layer](layer_input, h[layer], c[layer])

                # apply dropout to the output of each layer except the last
                if self.dropout is not None and layer < self.n_layers - 1:
                    layer_input = self.dropout(h[layer])
                else:
                    layer_input = h[layer]

        # use the final hidden state of the last layer
        out = self.fc(h[-1])
        return out



class LayerNormGRU(nn.Module):
    def __init__(self, input_size:int = 10, hidden_size:int = 20):
        super().__init__()

        self.hidden_size = hidden_size
        self.gates_layer = nn.Linear(input_size + hidden_size, 2 * hidden_size)
        self.gates_norm = nn.LayerNorm(2 * hidden_size)

        self.candidate_layer = nn.Linear(input_size + hidden_size, hidden_size)
        self.candidate_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, h_prev):

        # compute reset and update gates
        combined = torch.cat([x, h_prev], dim=-1)           # (batch, input+hidden)
        rz = self.gates_norm(self.gates_layer(combined))     # (batch, 2*hidden)
        r, z = rz.chunk(2, dim=-1)                          # each (batch, hidden)
        r, z = torch.sigmoid(r), torch.sigmoid(z)

        # compute candidate hidden state
        combined_r = torch.cat([x, r * h_prev], dim=-1)     # (batch, input+hidden)
        n = self.candidate_norm(self.candidate_layer(combined_r))  # (batch, hidden)
        n = torch.tanh(n)

        # compute new hidden state
        h = (1 - z) * h_prev + z * n
        return h
    



class MultiLayerNormGRU(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.cells = nn.ModuleList()
        for layer in range(n_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.cells.append(LayerNormGRU(layer_input_size, hidden_size))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.shape

        # initialize h to zeros for every layer
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.n_layers)]

        for t in range(seq_len):
            layer_input = x[:, t, :]

            for layer in range(self.n_layers):
                h[layer] = self.cells[layer](layer_input, h[layer])

                # apply dropout between layers, not after the last one
                if self.dropout is not None and layer < self.n_layers - 1:
                    layer_input = self.dropout(h[layer])
                else:
                    layer_input = h[layer]

        # use the final hidden state of the last layer
        out = self.fc(h[-1])
        return out
