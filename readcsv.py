import csv
def readin_csv(file):
    with open(file + ".csv", 'r') as f:
        reader = csv.reader(f)
        shape_sequence = []
        for i, record in enumerate(reader):
            if i == 0:
                continue
            landmarks = []
            for j in range(68):
                landmarks.append((eval(record[5 + j]), eval(record[5 + 68 + j])))
            shape_sequence.append(landmarks)
    return shape_sequence