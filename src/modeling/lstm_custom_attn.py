import torch
import torch.nn as nn
import torch.nn.functional as F


class LstmEncoderDecoderWithAttention(nn.Module):
    def __init__(
        self,
        vehicle_feature_sizes,
        embedding_dim,
        localization_input_size,
        control_input_size,
        hidden_size,
        num_layers,
        dropout=0.2,
    ):
        super(LstmEncoderDecoderWithAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # Vehicle feature embddings with Dropout
        self.vehicle_id_embedding = nn.Sequential(
            nn.Embedding(
                num_embeddings=vehicle_feature_sizes["vehicle_id"],
                embedding_dim=embedding_dim,
            ),
            nn.Dropout(p=dropout),
        )
        self.vehicle_model_embedding = nn.Sequential(
            nn.Embedding(
                num_embeddings=vehicle_feature_sizes["vehicle_model"],
                embedding_dim=embedding_dim,
            ),
            nn.Dropout(p=dropout),
        )
        self.vehicle_model_modification_embedding = nn.Sequential(
            nn.Embedding(
                num_embeddings=vehicle_feature_sizes["vehicle_model_modification"],
                embedding_dim=embedding_dim,
            ),
            nn.Dropout(p=dropout),
        )
        self.location_reference_point_id_embedding = nn.Sequential(
            nn.Embedding(
                num_embeddings=vehicle_feature_sizes["location_reference_point_id"],
                embedding_dim=embedding_dim,
            ),
            nn.Dropout(p=dropout),
        )
        self.tires_front_embedding = nn.Sequential(
            nn.Embedding(
                num_embeddings=vehicle_feature_sizes["tires_front"],
                embedding_dim=embedding_dim,
            ),
            nn.Dropout(p=dropout),
        )
        self.tires_rear_embedding = nn.Sequential(
            nn.Embedding(
                num_embeddings=vehicle_feature_sizes["tires_rear"],
                embedding_dim=embedding_dim,
            ),
            nn.Dropout(p=dropout),
        )

        self.vehicle_fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embedding_dim * 6, hidden_size),
        )

        self.attention = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.attention_combine = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size * 3, hidden_size),
        )

        self.localization_encoder = nn.LSTM(
            input_size=localization_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )

        # Encoder LSTM for input_control_seq
        self.control_encoder = nn.LSTM(
            input_size=control_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )

        self.decoder = nn.LSTM(
            input_size=control_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )

        self.fc_out = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, localization_input_size),
        )

    def forward(
        self,
        vehicle_features,
        input_localization,
        input_control_sequence,
        output_control_sequence,
    ):
        vehicle_id = self.vehicle_id_embedding(vehicle_features[:, 0])
        vehicle_model = self.vehicle_model_embedding(vehicle_features[:, 1])
        vehicle_model_modification = self.vehicle_model_modification_embedding(
            vehicle_features[:, 2]
        )
        location_reference_point_id = self.location_reference_point_id_embedding(
            vehicle_features[:, 3]
        )
        tires_front = self.tires_front_embedding(vehicle_features[:, 4])
        tires_rear = self.tires_rear_embedding(vehicle_features[:, 5])

        vehicle_embedded = torch.cat(
            [
                vehicle_id,
                vehicle_model,
                vehicle_model_modification,
                location_reference_point_id,
                tires_front,
                tires_rear,
            ],
            dim=1,
        )  # Shape: [batch_size, embedding_dim * 6]

        vehicle_features_encoded = self.vehicle_fc(
            vehicle_embedded
        )  # Shape: [batch_size, hidden_size]

        localization_output, (hidden_loc, cell_loc) = self.localization_encoder(
            input_localization
        )
        control_output, (hidden_ctrl, cell_ctrl) = self.control_encoder(
            input_control_sequence
        )
        hidden_enc = torch.cat(
            (hidden_loc[: self.num_layers], hidden_ctrl[: self.num_layers]), dim=2
        )

        # Combine cell states by averaging
        cell_enc = (cell_ctrl[: self.num_layers] + cell_loc[: self.num_layers]) / 2

        attention_weights = F.softmax(
            self.attention(hidden_enc), dim=2
        )  # [num_layers, batch_size, hidden_size]

        combined_hidden = torch.cat((hidden_enc, attention_weights), dim=2)
        combined_hidden = self.attention_combine(combined_hidden)

        combined_hidden = combined_hidden.clone()
        combined_hidden[0] = combined_hidden[0] + vehicle_features_encoded.unsqueeze(0)

        # Decoder
        decoder_output, (hidden_dec, cell_dec) = self.decoder(
            output_control_sequence, (combined_hidden, cell_enc)
        )

        # Output layer with Dropout
        output_localization = self.fc_out(
            decoder_output
        )  # [batch_size, seq_len, localization_input_size]
        return output_localization
