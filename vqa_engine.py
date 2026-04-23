import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

class VQAEngine:
    def __init__(self, model_id="microsoft/Phi-3.5-vision-instruct"):
        self.model_id = model_id
        self.model = None
        self.processor = None

    def check_dependencies(self):
        """
        Check if huggingface transformers and torch are accessible, which is trivial 
        here since we imported them, but keeps compatibility with the UI boot process.
        """
        try:
            import transformers
            import torch
        except ImportError as e:
            return False, f"Missing Python dependency: {str(e)}"

        return True, "Dependencies OK."

    def _load_model(self):
        """
        Lazily load the model so it doesn't freeze the UI on startup.
        Uses bfloat16 for reduced RAM footprint appropriate for Mac.
        """
        if self.model is None or self.processor is None:
            # Cache the model inside the project workspace directory
            cache_dir = os.path.join(os.path.dirname(__file__), "models")
            os.makedirs(cache_dir, exist_ok=True)
            
            # For Mac (MPS) or general CPU/GPU fallback, device_map="auto" usually works well.
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                cache_dir=cache_dir,
                device_map="auto",
                _attn_implementation='eager' 
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, 
                trust_remote_code=True, 
                num_crops=4,
                cache_dir=cache_dir
            )

    def get_answer_vision(self, image_path: str, question: str) -> str:
        """
        Builds the prompt and sends the image directly to the multimodal model.
        """
        try:
            self._load_model()
            
            image = Image.open(image_path)
            
            messages = [
                {"role": "user", "content": f"<|image_1|>\n{question}"}
            ]
            prompt = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Explicitly route the inputs to the model's device
            inputs = self.processor(prompt, [image], return_tensors="pt").to(self.model.device)
            
            generation_args = {
                "max_new_tokens": 512,
                "temperature": 0.0,
                "do_sample": False,
            }
            
            generate_ids = self.model.generate(
                **inputs, 
                eos_token_id=self.processor.tokenizer.eos_token_id, 
                **generation_args
            )
            
            # Remove input tokens from output to just get the generation
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = self.processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return response
            
        except Exception as e:
            raise RuntimeError(f"Error communicating with local Phi-3.5-vision model: {str(e)}")
