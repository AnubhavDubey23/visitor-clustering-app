from flask import Blueprint, render_template, request, redirect, send_file
import pandas as pd
from .utils import process_file
import os

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join('uploaded.csv')
            file.save(filepath)
            output_path = process_file(filepath)
            return send_file(output_path, as_attachment=True)
    return render_template('index.html')
