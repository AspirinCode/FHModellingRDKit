import os, webbrowser,pickle
import pandas as pd

def viewTable(table):
    # Create a web page view of the data for easy viewing
    html = table[0:99].to_html()

    # Save the html to a temporary file
    with open("data.html", "w") as f:
        f.write(html)

    # Open the web page in our web browser
    full_filename = os.path.abspath("data.html")
    webbrowser.open("file://{}".format(full_filename))

def pickledump(obj,filename):
    with open(filename, "w") as f:
       pickle.dump(obj, f)

def pickleload(filename):
    with open(filename, "r") as f:
        return pickle.load(f)