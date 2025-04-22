import os

def init():
  global project_path
  project_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
