import torch
import torch.nn as nn
import torch.nn.functional as F


class LstmEncoderDecoderWithLuongAttention(nn.Module):
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
        super(LstmEncoderDecoderWithLuongAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

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
            nn.Linear(embedding_dim * 6, hidden_size * 2),
        )

        self.localization_encoder = nn.LSTM(
            input_size=localization_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )

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
            hidden_size=hidden_size * 2,  # Adjusted for concatenated context
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )

        # Output layer with Dropout
        self.fc_out = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size * 2, localization_input_size),
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
        )  # Shape: [batch_size, hidden_size * 2]

        localization_output, (hidden_loc, cell_loc) = self.localization_encoder(
            input_localization
        )
        control_output, (hidden_ctrl, cell_ctrl) = self.control_encoder(
            input_control_sequence
        )

        encoder_outputs = torch.cat(
            (localization_output, control_output), dim=1
        )  # [batch_size, seq_len_enc, hidden_size]

        hidden_enc = torch.cat(
            (hidden_loc, hidden_ctrl), dim=2
        )  # [num_layers, batch_size, hidden_size * 2]
        cell_enc = torch.cat(
            (cell_loc, cell_ctrl), dim=2
        )  # [num_layers, batch_size, hidden_size * 2]

        hidden_dec = hidden_enc.clone()
        hidden_dec[0] = hidden_dec[0] + vehicle_features_encoded.unsqueeze(0)

        cell_dec = cell_enc

        hidden_size_dec = self.hidden_size * 2

        decoder_output, (hidden_dec, cell_dec) = self.decoder(
            output_control_sequence, (hidden_dec, cell_dec)
        )

        decoder_output_projected = decoder_output[
            :, :, : self.hidden_size
        ]  # [batch_size, seq_len_dec, hidden_size]

        attn_scores = torch.bmm(
            decoder_output_projected, encoder_outputs.transpose(1, 2)
        )  # [batch_size, seq_len_dec, seq_len_enc]

        attn_weights = F.softmax(attn_scores, dim=2)  # [batch_size, seq_len_dec, seq_len_enc]

        context = torch.bmm(
            attn_weights, encoder_outputs
        )  # [batch_size, seq_len_dec, hidden_size]

        combined = torch.cat(
            (decoder_output_projected, context), dim=2
        )  # [batch_size, seq_len_dec, hidden_size * 2]

        output_localization = self.fc_out(
            combined
        )  # [batch_size, seq_len_dec, localization_input_size]

        return output_localization
