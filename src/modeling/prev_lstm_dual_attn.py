# same as lstm dual attn, but w/o dropout
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoDPLstmEncoderDecoderWithDualAttention(nn.Module):
    def __init__(
        self,
        vehicle_feature_sizes,
        embedding_dim,
        localization_input_size,
        control_input_size,
        hidden_size,
        num_layers,
        dropout=0,
    ):
        super(NoDPLstmEncoderDecoderWithDualAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Vehicle feature embeddings
        self.vehicle_id_embedding = nn.Embedding(
            num_embeddings=vehicle_feature_sizes["vehicle_id"],
            embedding_dim=embedding_dim,
        )
        self.vehicle_model_embedding = nn.Embedding(
            num_embeddings=vehicle_feature_sizes["vehicle_model"],
            embedding_dim=embedding_dim,
        )
        self.vehicle_model_modification_embedding = nn.Embedding(
            num_embeddings=vehicle_feature_sizes["vehicle_model_modification"],
            embedding_dim=embedding_dim,
        )
        self.location_reference_point_id_embedding = nn.Embedding(
            num_embeddings=vehicle_feature_sizes["location_reference_point_id"],
            embedding_dim=embedding_dim,
        )
        self.tires_front_embedding = nn.Embedding(
            num_embeddings=vehicle_feature_sizes["tires_front"],
            embedding_dim=embedding_dim,
        )
        self.tires_rear_embedding = nn.Embedding(
            num_embeddings=vehicle_feature_sizes["tires_rear"],
            embedding_dim=embedding_dim,
        )

        # Fully connected layer to combine vehicle features
        self.vehicle_fc = nn.Linear(
            embedding_dim * 6, hidden_size * 2
        )  # Выходной размер: hidden_size * 2

        # Attention layers (Custom Attention)
        self.custom_attention = nn.Linear(hidden_size * 2, hidden_size)
        self.custom_attention_combine = nn.Linear(
            hidden_size * 3, hidden_size * 2
        )  # Выходной размер: hidden_size * 2

        # Encoder LSTM for input_localization_seq
        self.localization_encoder = nn.LSTM(
            input_size=localization_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        # Encoder LSTM for input_control_seq
        self.control_encoder = nn.LSTM(
            input_size=control_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=control_input_size,
            hidden_size=hidden_size * 2,  # Выходной размер: hidden_size * 2
            num_layers=num_layers,
            batch_first=True,
        )

        # Output layer
        self.fc_out = nn.Linear(
            hidden_size * 2, localization_input_size
        )  # Выходной размер: localization_input_size

    def forward(
        self,
        vehicle_features,
        input_localization,
        input_control_sequence,
        output_control_sequence,
    ):
        batch_size = vehicle_features.size(0)

        # Embed vehicle features
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

        # Concatenate vehicle features
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

        # Encode vehicle features
        vehicle_features_encoded = self.vehicle_fc(
            vehicle_embedded
        )  # Shape: [batch_size, hidden_size * 2]

        # Encoder for localization
        localization_output, (hidden_loc, cell_loc) = self.localization_encoder(
            input_localization
        )

        # Encoder for control sequence
        control_output, (hidden_ctrl, cell_ctrl) = self.control_encoder(
            input_control_sequence
        )

        # Combine encoder outputs
        encoder_outputs = torch.cat(
            (localization_output, control_output), dim=1
        )  # [batch_size, seq_len_enc*2, hidden_size]

        # Initial decoder hidden and cell states (concatenate)
        hidden_enc = torch.cat(
            (hidden_loc, hidden_ctrl), dim=2
        )  # [num_layers, batch_size, hidden_size * 2]

        cell_enc = torch.cat(
            (cell_loc, cell_ctrl), dim=2
        )  # [num_layers, batch_size, hidden_size * 2]

        # Custom Attention mechanism
        attention_weights = F.softmax(
            self.custom_attention(hidden_enc), dim=2
        )  # [num_layers, batch_size, hidden_size]

        combined_hidden = torch.cat(
            (hidden_enc, attention_weights), dim=2
        )  # [num_layers, batch_size, hidden_size * 3]

        combined_hidden = self.custom_attention_combine(
            combined_hidden
        )  # [num_layers, batch_size, hidden_size * 2]

        # Incorporate vehicle_features_encoded into hidden state (first layer)
        hidden_dec = hidden_enc.clone()
        hidden_dec[0] = (
            hidden_dec[0] + vehicle_features_encoded
        )  # [num_layers, batch_size, hidden_size * 2]

        cell_dec = cell_enc

        # Decoder
        decoder_output, (hidden_dec, cell_dec) = self.decoder(
            output_control_sequence, (hidden_dec, cell_dec)
        )
        # Luong Attention mechanism

        # Project decoder outputs to hidden_size if needed
        decoder_output_projected = decoder_output[
            :, :, : self.hidden_size
        ]  # [batch_size, seq_len_dec, hidden_size]

        # Compute attention scores (dot product)
        attn_scores = torch.bmm(
            decoder_output_projected, encoder_outputs.transpose(1, 2)
        )  # [batch_size, seq_len_dec, seq_len_enc*2]

        # Apply softmax to get attention weights
        attn_weights = F.softmax(
            attn_scores, dim=2
        )  # [batch_size, seq_len_dec, seq_len_enc*2]

        # Compute context vectors
        context = torch.bmm(
            attn_weights, encoder_outputs
        )  # [batch_size, seq_len_dec, hidden_size]

        # Concatenate decoder output and context
        combined = torch.cat(
            (decoder_output_projected, context), dim=2
        )  # [batch_size, seq_len_dec, hidden_size * 2]

        # Output layer
        output_localization = self.fc_out(
            combined
        )  # [batch_size, seq_len_dec, localization_input_size]

        return output_localization
