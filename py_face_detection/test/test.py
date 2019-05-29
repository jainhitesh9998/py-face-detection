import numpy

from proj_data.py_face_detection.embeddings.embeddings import emp
from proj_data.py_face_detection.embeddings import path as embeddings_path

print(emp["55550"])
# numpy.save(embeddings_path.get()+"/55550",emp["55550"])
print(numpy.load(embeddings_path.get('55550.npy')))
