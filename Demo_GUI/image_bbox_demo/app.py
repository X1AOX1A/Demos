import logging
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

class ImageUtils:
    def __init__(self):
        pass

    def read_image(self, uploaded_image):
        """Read the uploaded image"""
        raw_image = Image.open(uploaded_image)
        width, height = raw_image.size
        return raw_image, (width, height)

    def resize_image(self, raw_image, square=544):
        """Resize the mask so it fits inside a 544x544 square"""
        width, height = raw_image.size

        # Check if the image resolution is larger than
        if width > square or height > square:
            # Calculate the aspect ratio
            aspect_ratio = width / height

            # Calculate the new dimensions to fit within 960x544 frame
            if aspect_ratio > square / square:
                resized_width = square
                resized_height = int(resized_width / aspect_ratio)
            else:
                resized_height = square
                resized_width = int(resized_height * aspect_ratio)

            # Resize the image
            resized_image = raw_image.resize((resized_width, resized_height))
            return resized_image, (resized_width, resized_height)
        else:
            return raw_image, (width, height)


    def get_canvas(self, resized_image, key="canvas"):
        """Retrieves the canvas to receive the bounding boxes
        Args:
        resized_image(Image.Image): the resized uploaded image
        key(str): the key to initiate the canvas component in streamlit
        """
        width, height = resized_image.size
        st.write("Draw bounding boxes on the objects you want to caption on.")

        canvas_result = st_canvas(
            fill_color="rgba(255,0, 0, 0.1)",
            stroke_width=2,
            stroke_color="rgba(255,0,0,1)",
            background_color="rgba(0,0,0,1)",
            background_image=resized_image,
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode="rect",
            key=key,
        )
        return canvas_result

    def get_resized_boxes(self, canvas_result):
        """Get the resized boxes from the canvas result"""
        objects = canvas_result.json_data["objects"]
        resized_boxes = []
        for object in objects:
            left, top = object["left"], object["top"]           # upper left corner
            width, height = object["width"], object["height"]   # box width and height
            right, bottom = left + width, top + height          # lower right corner
            resized_boxes.append([left, top, right, bottom])
        return resized_boxes

    def get_raw_boxes(self, resized_boxes, raw_size, resized_size):
        """Convert the resized boxes to raw boxes"""
        raw_width, raw_height = raw_size
        resized_width, resized_height = resized_size
        raw_boxes = []
        for box in resized_boxes:
            left, top, right, bottom = box
            raw_left = int(left * raw_width / resized_width)
            raw_top = int(top * raw_height / resized_height)
            raw_right = int(right * raw_width / resized_width)
            raw_bottom = int(bottom * raw_height / resized_height)
            raw_boxes.append([raw_left, raw_top, raw_right, raw_bottom])
        return raw_boxes

def main():
    st.set_page_config(layout="wide")
    logging.basicConfig(
        level = logging.ERROR,
        format = "%(asctime)s%(levelname)s%(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S"
    )

    uploaded_image=st.file_uploader("Upload image here: ", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_image is not None:
        utils = ImageUtils()
        raw_image, raw_size = utils.read_image(uploaded_image)
        resized_image, resized_size = utils.resize_image(raw_image)

        col1, col2 = st.columns(2)
        with col1:
            # read bbox input
            canvas_result = utils.get_canvas(resized_image)
            # read text input
            default_text = "Write a description for the photo."
            input_text = st.text_input("Text here:", value=default_text)
            submit = st.button("Submit")
        with col2:
            if submit:
                resized_boxes = utils.get_resized_boxes(canvas_result)
                # left_upper point and right_lower point : [x1, y1, x2, y2]
                raw_boxes = utils.get_raw_boxes(resized_boxes, raw_size, resized_size)
                sample = {
                    "image": raw_image,
                    "bboxes": raw_boxes,
                    "input_text": input_text,
                }
                st.write("Received the following sample:")
                st.success(sample)

if __name__ == "__main__":
    main()

# streamlit run app.py --server.fileWatcherType none