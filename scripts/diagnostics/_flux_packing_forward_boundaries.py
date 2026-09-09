"""Passive observations around the first FLUX block and final output norm."""

from contextlib import contextmanager


FIRST_BLOCK_STAGES = (
    "transformer_blocks.0.input",
    "transformer_blocks.0.attention_input",
    "transformer_blocks.0.attention_output",
)
OUTPUT_STAGES = ("norm_out_input", "norm_out")


def boundary_stage_order(block_names):
    """Place boundary observations in their actual forward execution order."""
    names = list(block_names)
    if not names or names[0] != "transformer_blocks.0":
        raise ValueError("boundary capture requires transformer_blocks.0 first")
    return [*FIRST_BLOCK_STAGES, *names, *OUTPUT_STAGES, "velocity"]


def _argument(args, kwargs, index, name):
    return args[index] if len(args) > index else kwargs[name]


@contextmanager
def boundary_hooks(model, capture, split_block_output):
    """Observe existing modules without changing their inputs or outputs.

    ``capture.record`` owns snapshots and comparisons. ``conditioning`` retains
    the per-example timestep embedding supplied to the final adaptive norm.
    """
    if not model.transformer_blocks:
        raise ValueError("boundary capture requires at least one double block")
    block = model.transformer_blocks[0]
    count = len(capture.lengths)

    def double_parts(text, image):
        return split_block_output("double", (text, image), capture.lengths, capture.image_tokens)

    def block_input(module, args, kwargs):
        image = _argument(args, kwargs, 0, "hidden_states")
        text = _argument(args, kwargs, 1, "encoder_hidden_states")
        capture.record(FIRST_BLOCK_STAGES[0], double_parts(text, image))

    def attention_input(module, args, kwargs):
        image = _argument(args, kwargs, 0, "hidden_states")
        text = _argument(args, kwargs, 1, "encoder_hidden_states")
        capture.record(FIRST_BLOCK_STAGES[1], double_parts(text, image))

    def attention_output(module, args, kwargs, output):
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError("double attention must return (image, text)")
        image, text = output
        capture.record(FIRST_BLOCK_STAGES[2], double_parts(text, image))

    def image_parts(image):
        if image.ndim != 3 or image.shape[:2] != (count, capture.image_tokens):
            raise ValueError("unexpected output norm image shape")
        return [{"image": image[i]} for i in range(count)]

    def norm_input(module, args, kwargs):
        image = _argument(args, kwargs, 0, "x")
        conditioning = _argument(args, kwargs, 1, "conditioning_embedding")
        parts = image_parts(image)
        if conditioning.ndim != 2 or conditioning.shape[0] != count:
            raise ValueError("unexpected output norm conditioning shape")
        for i, part in enumerate(parts):
            part["conditioning"] = conditioning[i]
        capture.record(OUTPUT_STAGES[0], parts)

    def norm_output(module, args, kwargs, output):
        capture.record(OUTPUT_STAGES[1], image_parts(output))

    handles = []
    try:
        handles.append(block.register_forward_pre_hook(block_input, with_kwargs=True))
        handles.append(block.attn.register_forward_pre_hook(attention_input, with_kwargs=True))
        handles.append(block.attn.register_forward_hook(attention_output, with_kwargs=True))
        handles.append(model.norm_out.register_forward_pre_hook(norm_input, with_kwargs=True))
        handles.append(model.norm_out.register_forward_hook(norm_output, with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()
