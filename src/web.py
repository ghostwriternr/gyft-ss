from flask import Flask, url_for, send_file, send_from_directory, request, Response, jsonify
from flask_cors import CORS
import logging
import os
import time
from gyftss import convert

app = Flask(__name__)
CORS(app)

file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.DEBUG)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/input_images'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath


def allowed_file(filename):
    return '.' in filename and filename.rsplit(
        '.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    """
    Handle http request to root
    """
    return "Hello!"


@app.route('/convert', methods=['POST'])
def api_root():
    app.logger.info(PROJECT_HOME)
    # try:
    img = request.files['image']
    print(img)
    if request.method == 'POST' and img and allowed_file(img.filename):
        app.logger.info(app.config['UPLOAD_FOLDER'])
        img = request.files['image']
        create_new_folder(app.config['UPLOAD_FOLDER'])
        file_name = str(time.time()).replace(
            '.', '') + "_" + img.filename.replace('/', '')
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        app.logger.info("saving {}".format(saved_path))
        img.save(saved_path)
        timetable = convert(saved_path)
        timetable_arr = []
        for row in timetable:
            timetable_row = []
            for cell in row:
                timetable_row.append(cell.decode("utf-8"))
            timetable_arr.append(timetable_row)
        return jsonify(timetable=timetable_arr)
    else:
        return Response("No image sent", status=401)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080, use_reloader=False)
