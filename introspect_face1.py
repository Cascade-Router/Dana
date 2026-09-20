import FreeCAD as App
import Part

doc = App.ActiveDocument
if doc is None:
    doc = App.newDocument("TestDoc")

obj = doc.getObject("test_box")
if obj is None:
    obj = doc.addObject("Part::Box", "test_box")
    doc.recompute()

face = obj.Shape.Faces[0]  # Face1

print("dir(face.Surface.Position):")
print(dir(face.Surface.Position))

v_x = App.Vector(1, 0, 0)
v_y = App.Vector(0, 1, 0)

print("Rotation.multVec(App.Vector(1, 0, 0)):")
print(face.Surface.Position.Rotation.multVec(v_x))

print("Rotation.multVec(App.Vector(0, 1, 0)):")
print(face.Surface.Position.Rotation.multVec(v_y))
