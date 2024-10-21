from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from predict import *
from cv2 import *
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/ducnguyen/Desktop/duc doc/hackatum-allianz/segmentation_models.pytorch')

# PEOPLE_FOLDER = os.path.join('static', 'people_photo')
# app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER


app = Flask(__name__)


@app.route('/')
def upload_file():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploader_file():
    if request.method == 'POST':
        f = request.files['file']
        full_filename = os.path.join("static", secure_filename(f.filename))
        f.save(full_filename)

        img = cv2.imread(full_filename)
        img = cv2.resize(img, dsize=(512, 320))

        cv2.imwrite(full_filename, img)

        result_file_path = ''.join(full_filename.split('.')[:-1])+'result.png'
        result = predict(img)

        plt.gca().set_axis_off()
        plt.imshow(result)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)

        plt.savefig(result_file_path, transparent=True, bbox_inches='tight', pad_inches = 0)

        return render_template('upload.html', user_image=full_filename,
                               result_image=result_file_path)


if __name__ == '__main__':
    app.run(debug=True)
