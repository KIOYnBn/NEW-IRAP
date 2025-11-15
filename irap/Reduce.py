import os
import sys
import time
now_path = os.getcwd()

def visual_timestamp():
    a = time.mktime(time.localtime())
    return str(int(a))

def load_fasta_file(file):
    with open(file, 'r') as f:
        data = f.readlines()[-1]
    return data.strip()

def reduce_sq(sq, raalist):
    new_sq = ''
    for i in sq:
        for j in raalist:
            if i in j:
                new_sq += j[0]
    return new_sq


def plot_reduce_xy(aa, x, y, color):
    out = '<rect width="18" height="22" x="' + str(x - 9) + '" y="' + str(y - 11) + '" fill="' + color[aa] + '"></rect><text fill="white" x="' + str(x) + '" y="' + str(y) + '" text-anchor="middle" dy="6">' + aa + '</text><text fill="#333333" x="' + str(x) + '" y="' + str(y + 22) + '" text-anchor="middle" dy="5">|</text>'
    return out


def plot_reduce_rexy(aa, x, y, color):
    out = '<rect width="18" height="22" x="' + str(x - 9) + '" y="' + str(y + 33) + '" fill="' + color[aa] + '"></rect><text fill="white" x="' + str(x) + '" y="' + str(y + 44) + '" text-anchor="middle" dy="6">' + aa + '</text>'
    return out


def reduce_svg(sq, resq, out):
    head = '<svg xmlns="http://www.w3.org/2000/svg" width="1924" height="' + str(140*int(1+len(sq)/50)) + '">'
    end = '</svg>'
    color = {'A':'#f3a683', 'R':'#f7d794', 'N':'#778beb', 'D':'#e77f67', 'C':'#cf6a87', 'Q':'#f19066', 'E':'#f5cd79', 'G':'#546de5', 'H':'#e15f41', 'I':'#c44569', 'L':'#786fa6', 'K':'#f8a5c2', 'M':'#63cdda', 'F':'#ea8685', 'P':'#596275', 'S':'#574b90', 'T':'#f78fb3', 'W':'#3dc1d3', 'Y':'#e66767', 'V':'#303952'}
    y = 16
    mid = ''
    for i in range(int(len(sq)/50)+1):
        eachsq = sq[i*50:(i+1)*50]
        eachresq = resq[i*50:(i+1)*50]
        x = 106
        each_natural = '<text fill="#333333" x="5" y="' + str(y) + '" dy="6">Natural   ' + str(i*50) + '</text>'
        mid += each_natural
        for j in range(len(eachsq)):
            x += 18
            mid += plot_reduce_xy(eachsq[j], x, y, color)
        x = 106
        each_reduce = '<text fill="#333333" x="5" y="' + str(y + 44) + '" dy="6">Reduced   ' + str(i*50) + '</text>'
        mid += each_reduce
        for j in range(len(eachsq)):
            x += 18
            mid += plot_reduce_rexy(eachresq[j], x, y, color)
        y += 88
    all_data = head + mid + end
    with open(out + 'reduce.svg', 'w') as f:
        f.write(all_data)


def reduce_save(sq, resq, out):
    with open(out + 'sq.txt', 'w') as f:
        f.write('>' + out + 'Natural\n' + sq + '\n>' + out + 'Reduce\n' + resq)
        

def reduce(file=None, sq=None, out=now_path, raa=None):
    # get sequence
    if sq == None:
        sq = load_fasta_file(file)
    else:
        sq = sq
    # get raalist
    if raa != None:
        raalist = raa.split('-')
    else:
        raalist = 'LVIMCAGSTPFYW-EDNQKRH'.split('-')
    # get reduce sequence
    resq = reduce_sq(sq, raalist)
    # save
    if out == now_path:
        now_time = visual_timestamp()
        out = os.path.join(now_path, now_time + '-')
    # plot
    reduce_svg(sq, resq, out)
    # save
    reduce_save(sq, resq, out)

if __name__ == '__main__':
    requests = sys.argv[1:]
    file, raa, out = requests[0], requests[1], requests[2]
    reduce(file=file, out=out, raa=raa)
