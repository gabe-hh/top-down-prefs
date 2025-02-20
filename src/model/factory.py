import yaml
import torch
import torch.nn.functional as F
import os

from src.model.encoder import ConvEncoder, ConvEncoderCategorical, DenseEncoder, DenseEncoderCategorical
from src.model.decoder import ConvDecoder, DenseDecoder
from src.model.transition import RSSMTransitionCategorical, RNNTransitionCategorical, TransitionNormal, TransitionCategorical
from src.model.world_model import WorldModel
from src.model.latent_action import LatentActionModel
from src.utils.latent_handler import GaussianLatentHandler, CategoricalLatentHandler

# A simple mapping from string names to classes
ENCODER_MAP = {
    "ConvEncoder": ConvEncoder,
    "ConvEncoderCategorical": ConvEncoderCategorical,
    "DenseEncoder": DenseEncoder,
    "DenseEncoderCategorical": DenseEncoderCategorical,
}

DECODER_MAP = {
    "ConvDecoder": ConvDecoder,
    "DenseDecoder": DenseDecoder,
}

TRANSITION_MAP = {
    "RSSMTransitionCategorical": RSSMTransitionCategorical,
    "RNNTransitionCategorical": RNNTransitionCategorical,
    "TransitionNormal": TransitionNormal,
    "TransitionCategorical": TransitionCategorical,
}

LATENT_HANDLER_MAP = {
    "GaussianLatentHandler": GaussianLatentHandler,
    "CategoricalLatentHandler": CategoricalLatentHandler,
}

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def build_latent_action_model_from_config(config):
    model_config = config["model"]

    state_dim = model_config["state_dim"]
    action_dim = model_config["action_dim"]
    encoder_layers = model_config["encoder_layers"]
    decoder_layers = model_config["decoder_layers"]
    latent_handler_type = model_config["latent_handler"]["type"]
    LatentHandlerClass = LATENT_HANDLER_MAP[latent_handler_type]
    latent_handler_params = model_config["latent_handler"]
    latent_handler_params.pop("type")
    latent_handler = LatentHandlerClass(**latent_handler_params)
    deterministic_dim = model_config.get("deterministic_dim", 0)
    predict_deterministic = model_config.get("predict_deterministic", False)
    include_initial_deterministic = model_config.get("include_initial_deterministic", False)
    action_classes = model_config.get("action_classes", None)
    num_classes = model_config.get("num_classes", None)

    model = LatentActionModel(state_dim, action_dim, encoder_layers, decoder_layers, latent_handler, 
                              deterministic_dim=deterministic_dim, 
                              predict_deterministic=predict_deterministic, 
                              include_initial_deterministic=include_initial_deterministic, 
                              action_classes=action_classes, 
                              num_classes=num_classes)
    return model

def build_world_model_from_config(config):
    model_config = config["model"]

    # Build the encoder
    encoder_type = model_config["encoder"]["type"]
    EncoderClass = ENCODER_MAP[encoder_type]
    encoder_params = model_config["encoder"]
    # Remove the type field since it is not a parameter
    encoder_params.pop("type")
    encoder = EncoderClass(**encoder_params)

    # Build the decoder
    decoder_type = model_config["decoder"]["type"]
    DecoderClass = DECODER_MAP[decoder_type]
    decoder_params = model_config["decoder"]
    decoder_params.pop("type")
    decoder = DecoderClass(**decoder_params)

    # Build the transition module
    transition_type = model_config["transition"]["type"]
    TransitionClass = TRANSITION_MAP[transition_type]
    transition_params = model_config["transition"]
    transition_params.pop("type")
    transition = TransitionClass(**transition_params)

    # Build the latent handler
    latent_handler_type = model_config["latent_handler"]["type"]
    LatentHandlerClass = LATENT_HANDLER_MAP[latent_handler_type]
    latent_handler_params = model_config["latent_handler"]
    latent_handler_params.pop("type")
    latent_handler = LatentHandlerClass(**latent_handler_params)

    # Build the overall model
    steps = model_config.get("steps", 5)
    reset_hidden = model_config.get("reset_hidden", True)
    model = WorldModel(encoder, decoder, transition, latent_handler, (encoder_params["latent_dim"], encoder_params["num_classes"]), steps=steps, reset_hidden=reset_hidden)
    return model

def build_model_from_config(config_path, device="cuda"):
    config = load_config(config_path)
    return build_model_from_loaded_config(config, device)

def build_model_from_loaded_config(config, device="cuda"):
    model_type = config["model"]["type"]
    if model_type == "ModelLow" or model_type == "ModelHigh":
        model = build_world_model_from_config(config)
    elif model_type == "LatentAction":
        model = build_latent_action_model_from_config(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model.to(device)

def load_state_dict_from_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    return model

def load_model(model_root, checkpoint="best", device="cuda"):
    config_path = os.path.join(model_root, "config.yaml")
    model = build_model_from_config(config_path, device)
    checkpoint_path = os.path.join(model_root, f"{checkpoint}.pt")
    model = load_state_dict_from_checkpoint(model, checkpoint_path)
    return model.to(device)

# Example usage:
if __name__ == "__main__":
    config = load_config("model_config.yaml")
    model = build_world_model_from_config(config)
    print(model)
