IMAGE_PLACEHOLDER_TOKEN = "<image_placeholder>"


def get_prompt_parts(
    prompt: str, image_token: str = IMAGE_PLACEHOLDER_TOKEN
) -> list[str]:
    prompt_parts = prompt.split(image_token)
    if len(prompt_parts) == 1:
        prompt_parts = [""] + prompt_parts

    return prompt_parts
