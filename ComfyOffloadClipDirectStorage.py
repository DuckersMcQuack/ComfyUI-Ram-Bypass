import types
import torch
import mmap
import os
import comfy.model_management

CACHE_PATH = "D:/Stablediff/Comfyuimanual/ComfyUI/models/directstorage/clipmodel.pt"  # Change to your NVMe path

class OverrideDevice:
    @classmethod
    def INPUT_TYPES(s):
        devices = ["cpu",]
        for k in range(0, torch.cuda.device_count()):
            devices.append(f"cuda:{k}")
        return {"required": {"device": (devices, {"default":"cpu"})}}
    
    FUNCTION = "patch"
    CATEGORY = "other"

    def override(self, model, model_attr, device):
        model.device = device
        patcher = getattr(model, "patcher", model)
        for name in ["device", "load_device", "offload_device", "current_device", "output_device"]:
            setattr(patcher, name, device)
        
        py_model = getattr(model, model_attr)
        py_model.to = types.MethodType(torch.nn.Module.to, py_model)
        py_model.to(device)
        
        def to(*args, **kwargs):
            pass
        py_model.to = types.MethodType(to, py_model)
        return (model,)

    def patch(self, *args, **kwargs):
        raise NotImplementedError

class OverrideCLIPDevice(OverrideDevice):
    @classmethod
    def INPUT_TYPES(s):
        k = super().INPUT_TYPES()
        k["required"]["clip"] = ("CLIP",)
        return k
    
    RETURN_TYPES = ("CLIP",)
    TITLE = "Force/Set CLIP Device with NVMe Cache"

    def patch(self, clip, device):
        device = torch.device(device)
        
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "r+b") as f:
                mmapped_model = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                model_data = torch.load(mmapped_model, map_location=device)
                clip.cond_stage_model.load_state_dict(model_data)
                mmapped_model.close()
        else:
            model_data = clip.cond_stage_model.state_dict()
            with open(CACHE_PATH, "wb") as f:
                torch.save(model_data, f)
        
        return self.override(clip, "cond_stage_model", device)

class OverrideVAEDevice(OverrideDevice):
    @classmethod
    def INPUT_TYPES(s):
        k = super().INPUT_TYPES()
        k["required"]["vae"] = ("VAE",)
        return k
    
    RETURN_TYPES = ("VAE",)
    TITLE = "Force/Set VAE Device"

    def patch(self, vae, device):
        return self.override(vae, "first_stage_model", torch.device(device))

NODE_CLASS_MAPPINGS = {
    "OverrideCLIPDevice": OverrideCLIPDevice,
    "OverrideVAEDevice": OverrideVAEDevice,
}
NODE_DISPLAY_NAME_MAPPINGS = {k:v.TITLE for k,v in NODE_CLASS_MAPPINGS.items()}
