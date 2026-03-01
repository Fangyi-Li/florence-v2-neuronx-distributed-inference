#!/usr/bin/env python3
"""
Separate model loader to avoid import-time issues with config patching.
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


def load_florence2_model(model_name: str):
    """
    Load Florence-2 model with compatibility fixes.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        Tuple of (model, processor)
    """
    # Patch the config class before loading
    from transformers.models.auto import configuration_auto
    from transformers import AutoConfig
    
    # Load config first to trigger download
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except AttributeError as e:
        if "forced_bos_token_id" in str(e):
            # The config has the bug, we need to patch it
            print("Detected forced_bos_token_id issue, applying workaround...")
            
            # Import the config module dynamically
            import sys
            import importlib
            
            # Find and patch the Florence2LanguageConfig class
            for module_name in list(sys.modules.keys()):
                if 'configuration_florence2' in module_name:
                    module = sys.modules[module_name]
                    if hasattr(module, 'Florence2LanguageConfig'):
                        config_class = module.Florence2LanguageConfig
                        original_init = config_class.__init__
                        
                        def patched_init(self, *args, **kwargs):
                            # Set defaults before calling original
                            self.__dict__['forced_bos_token_id'] = kwargs.pop('forced_bos_token_id', None)
                            self.__dict__['forced_eos_token_id'] = kwargs.pop('forced_eos_token_id', None)
                            original_init(self, *args, **kwargs)
                        
                        config_class.__init__ = patched_init
                        print(f"Patched {config_class.__name__}")
                        break
            
            # Try loading again
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        else:
            raise
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    )
    model.eval()
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    return model, processor

