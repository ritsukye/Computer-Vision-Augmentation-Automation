import albumentations as A

def build_aug(name, cfg):
    if name == "hue_shift":
        return A.HueSaturationValue(hue_shift_limit=cfg["limit"], p=1)

    if name == "saturation_shift":
        return A.ColorJitter(saturation=cfg["limit"], p=1)

    if name == "brightness_shift":
        return A.ColorJitter(brightness=cfg["limit"], p=1)

    if name == "contrast_shift":
        return A.ColorJitter(contrast=cfg["limit"], p=1)

    if name == "gamma_shift":
        return A.RandomGamma(gamma_limit=cfg["gamma_limit"], p=1)

    if name == "rgb_shift":
        return A.RGBShift(
            r_shift_limit=cfg["r_shift"],
            g_shift_limit=cfg["g_shift"],
            b_shift_limit=cfg["b_shift"],
            p=1,
        )

    if name == "clahe":
        return A.CLAHE(clip_limit=cfg["clip_limit"], p=1)

    if name == "white_balance_shift":
        t = cfg["temperature"]
        return A.RGBShift(
            r_shift_limit=t,
            g_shift_limit=0,
            b_shift_limit=t,
            p=1,
        )

    if name == "gaussian_noise":
        var_min, var_max = cfg["var_limit"]
        return A.GaussNoise(
            std_range=(var_min ** 0.5/225, var_max ** 0.5/225),
            p=1
        )

    if name == "motion_blur":
        return A.MotionBlur(blur_limit=cfg["blur_limit"], p=1)

    if name == "gaussian_noise":
        return A.GaussNoise(
            var_limit=tuple(cfg["var_limit"]),
            p=1
        )

    if name == "jpeg_compress":
        return A.ImageCompression(
            quality_range=tuple(cfg["quality"]),
            p=1,
        )

    if name == "cutout":
        holes = cfg["holes"]
        return A.CoarseDropout(
            num_holes_range=(holes, holes),
            hole_height_range=(64, 64),
            hole_width_range=(64, 64),
            p=1,
        )

    if name == "resize_degrade":
        return A.Downscale(
            scale_range=(cfg["min_scale"], 1.0),
            p=1
        )

    if name == "rotate":
        return A.Rotate(limit=cfg["limit"], p=1)

    if name == "crop":
        return A.RandomResizedCrop(
            size=(256, 256),
            scale=tuple(cfg["scale"]),
            p=1,
        )

    if name == "zoom":
        return A.Affine(scale=(1 - cfg["limit"], 1 + cfg["limit"]), p=1)

    if name == "shift":
        return A.Affine(translate_percent=cfg["limit"], p=1)

    if name == "shear":
        return A.Affine(shear=cfg["limit"], p=1)

    if name == "perspective":
        return A.Perspective(scale=cfg["scale"], p=1)

    if name == "flip":
        return A.HorizontalFlip(p=1)

    return None