python


class ImageToVideoConverter:
    def __init__(self, input_folder='./image', output_file='./output.mp4', frame_size=(960, 540), fps=30):
        self.input_folder = input_folder
        self.output_file = output_file
        self.frame_size = frame_size
        self.fps = fps

    def convert(self):
        image_extensions = ["*.png", "*.PNG", "*.JPG", "*.JPEG", "*.jpg", "*.jpeg", "*.bmp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.input_folder, ext)))
        image_files.sort()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_file, fourcc, self.fps, self.frame_size)

        for image_file in image_files:
            img = cv2.imread(image_file)
            img_resized = cv2.resize(img, self.frame_size)
            out.write(img_resized)

        out.release()
        cv2.destroyAllWindows()


converter = ImageToVideoConverter(input_folder='./images', output_file='./output.mp4')
converter.convert()
