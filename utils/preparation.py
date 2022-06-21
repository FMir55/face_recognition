

def get_suspects(path):
  suspects = []
  #check passed db folder exists
  if os.path.isdir(path) == True:
    for r, d, f in os.walk(path): # r=root, d=directories, f = files
      for file in f:
        if ('.jpg' in file):
          #exact_path = os.path.join(r, file)
          exact_path = r + "/" + file
          #print(exact_path)
          suspects.append(exact_path)
  if len(suspects) == 0:
    print("WARNING: There is no image in this path ( ", path_face_db,") . Face recognition will not be performed.")
  return suspects
