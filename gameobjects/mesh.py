#!/usr/bin/env python3

# This was present in one of the older versions of
# GameObjects

from OpenGL.GL import *
from OpenGL.GLU import *

class Mesh:
    
    def __init__(self):
        
        self.vertices = []
        self.texture_coords = []
        self.vertex_normals = []
        
        self.faces = []
        
    def read_obj(self, filename):
        "Read object"
        with open(filename, 'r', encoding='utf-8') as obj_file:
            file_in = obj_file.read().splitlines()
            obj_file.close()
        
        for line in file_in:
            words = line.split()
            
            if not words:
                continue
            
            data_type = words[0]
            data = words[1:]
            
            if data_type == '#': # Comment
                continue
            
            if data_type == 'v': # Vertex
                # v x y z
                vertex = float(data[0]), float(data[1]), float(data[2])
                self.vertices.append(vertex)
            
            elif data_type == 'vt': # Texture coordinate
                #vt u v
                texture_coord = float(data[0]), float(data[1])
                self.vertices.append(texture_coord)
            
            elif data_type == 'vn': # Vertex normal
                #vn x y z
                normal = float(data[0]), float(data[1]), float(data[2])
                self.vertex_normals.append(normal)
            
            elif data_type == 'f': # Face
##                assert len(data) == 3, 'Only triangles!'
                #f vertex_index, texture_index, normal_index
                for word in data:
                    triplet = word.split('/')
                    if len(triplet) == 2:
                        triplet = triplet[0], 0, triplet[1]
                    face = int(triplet[0]), int(triplet[1]), int(triplet[2])
                    self.faces.append(face)
    
    def draw(self):
        glBegin(GL_TRIS)
        
        for vertex_index, texture_index, normal_index in self.faces:
            glTexCoord(self.texture_coords[texture_index])
            glNormal(self.normals[normal_index])
            glVertex(vertices[vertex_index])
        
        glEnd()
        
if __name__ == '__main__':
    mesh = Mesh()
####    mesh.read_obj('will31.obj')
##    mesh.read_obj('models/mytank.obj')
    print(mesh.vertices)
