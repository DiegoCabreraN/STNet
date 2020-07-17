import os
from flask import Flask, render_template, url_for, request, flash, redirect
from werkzeug.utils import secure_filename
from model import run
import time

UPLOAD_PATH = "./static/uploads"
ALLOWED_EXTENSIONS = {'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_PATH'] = UPLOAD_PATH

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Upload', methods=['GET', 'POST'])
def uploads():
    if request.method == 'GET':
        return render_template('upload.html')
    else:
        if 'content' not in request.files:
            flash('No content image selected')
            return redirect(request.url)
        content = request.files['content']

        if 'style' not in request.files:
            flash('No style image selected')
            return redirect(request.url)
        style = request.files['style']

        if content.filename == '' or style.filename == '':
            flash('Error, please, try again')
            return redirect(request.url)

        if style and allowed_file(style.filename) and \
            content and allowed_file(content.filename):

            style_path = os.path.join(app.config['UPLOAD_PATH'], 'style.jpeg')
            style.save(style_path)
            content_path = os.path.join(app.config['UPLOAD_PATH'], 'content.jpeg')
            content.save(content_path)
            return run()
            

@app.route('/Result')
def process():
    return render_template('result.html')


if __name__ == '__main__':
    app.run(debug=True)