import gc
import sys
import torch
import argparse
import gradio as gr

try:
    from loguru import logger as logging
    logging.add(sys.stderr, filter="my_module")
except ImportError:
    import logging

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def load_model(args):
    raise NotImplementedError("Please implement the load_model function in app.py")


if __name__ == "__main__":    
    args = parse_args()
    
    logging.info('Initializing Model...')
    # prepare the model
    model = load_model(args)

    def submit(raw_image, text_input):
        try:
            logging.info("Received text input: {}".format(text_input))
            logging.info('Generating...')
            samples = {
                "image": raw_image,
                "text_input": text_input,
            }
            output = model.generate(samples)[0]
            logging.info('Generated: {}'.format(output))
            return output
        except Exception as e:
            return "An error occurred: {}".format(str(e))
        finally:
            del image, samples
            gc.collect()
            torch.cuda.empty_cache()

    demo = gr.Interface(
        fn=submit,
        inputs=[gr.Image(type="pil"),
                gr.Textbox(lines=1, placeholder="Text input here...")],
        outputs=["text"],
        flagging_options=["successed", "failed"],
        title="Image Captioning Demo",
        description="This is a demo for the Image Captioning model. Please input an image and a text.",
    )
    demo.launch(share=True, show_error=True)

# python app.py --cfg-path path/to/config  --gpu-id 0