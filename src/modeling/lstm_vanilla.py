import torch
import torch.nn as nn


class LstmEncoderDecoder(nn.Module):
    def __init__(
        self,
        vehicle_feature_sizes,
        embedding_dim,
        localization_input_size,
        control_input_size,
        hidden_size,
        num_layers,
    ):
        super(LstmEncoderDecoder, self).__init__()

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
        self.vehicle_fc = nn.Linear(embedding_dim * 6, hidden_size)

        # Encoder LSTM for input_localization_seq
        self.localization_encoder = nn.LSTM(
            input_size=localization_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Encoder LSTM for input_control_seq
        self.control_encoder = nn.LSTM(
            input_size=control_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=control_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc_out = nn.Linear(hidden_size, localization_input_size)

    def forward(
        self,
        vehicle_features,
        input_localization,
        input_control_sequence,
        output_control_sequence,
    ):
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
        vehicle_features_encoded = self.vehicle_fc(
            vehicle_embedded
        )  # Shape: [batch_size, hidden_size]

        # Encoder for localization
        _, (hidden_loc, cell_loc) = self.localization_encoder(
            input_localization
        )  # hidden_loc: [num_layers, batch_size, hidden_size]

        # Encoder for control sequence
        _, (hidden_ctrl, cell_ctrl) = self.control_encoder(
            input_control_sequence
        )  # hidden_ctrl: [num_layers, batch_size, hidden_size]

        # Combine encoder hidden states and vehicle features
        # Option to concatenate, sum, or average hidden states
        hidden_enc = (
            hidden_loc + hidden_ctrl
        ) / 2  # Shape: [num_layers, batch_size, hidden_size]
        cell_enc = (cell_loc + cell_ctrl) / 2

        # Incorporate vehicle features into the hidden state
        # We'll add vehicle_features_encoded to the first layer's hidden state
        hidden_enc[0] = hidden_enc[0] + vehicle_features_encoded.unsqueeze(0)

        # Decoder
        decoder_output, _ = self.decoder(
            output_control_sequence, (hidden_enc, cell_enc)
        )  # decoder_output: [batch_size, seq_len, hidden_size]

        # Output layer
        output_localization = self.fc_out(
            decoder_output
        )  # Shape: [batch_size, seq_len, localization_input_size]

        return output_localization
