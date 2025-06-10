from llamafactory.data.converter import DatasetConverter, register_dataset_converter


class Im2SvgConverter(DatasetConverter):
    """Takes a dataset of `("Svg":str,"image":PIL.Image.Image,"width":float,"height":float)` and converts it into a dataset of `("prompt":list[dict],"response":list[dict],"image":list[str])`."""

    def __call__(self, example):
        # read the image
        imgs = self._find_medias(example[self.dataset_attr.images])
        width = example["width"]
        height = example["height"]
        # build our chat‐style prompt/response
        prompt = [
            {
                "role": "user",
                "content": (
                    f"You first analyze the input image, think about how to convert it into SVG format, "
                    f"then generate SVG code that would render the image exactly as you see it. "
                    f"Think about the key shapes, paths, and visual elements that need to be represented and their x/y coordinates "
                    f"and any nesting or transformations necessary. "
                    f"Please recreate this image: <image> as an svg of width {width: 0.0f} and height {height: 0.0f}."
                ),
            }
        ]
        response = [{"role": "assistant", "content": example[self.dataset_attr.response]}]

        return {
            "_prompt": prompt,
            "_response": response,
            "_system": "",
            "_tools": "",
            "_images": imgs,
            "_videos": None,
            "_audios": None,
        }


# give it a name, e.g. "im2svg"
register_dataset_converter("im2svg", Im2SvgConverter)
