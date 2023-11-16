import tarfile
import os 
from pathlib import Path
import json

""" Makes sources for dataloader tests.
Running main will...
- generate 2 sources, titled 'source_id_00', 'source_id_01'
- each source has 7 .tar files, each with 100 sequences (except the last which has 66)
- each sequence has the first three tokens as (source_num, tar_num, line_num)

This way we'll be able to identify where each sequence came from when we test...
"""
	

def make_tar(tar_num, num_lines, source_num=0, dir_name=None):
    fname = lambda i: '%08d_chunk_%s.json' % (tar_num, i)
    
    if dir_name != None:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        
    tarname = os.path.join(dir_name, '%08d.tar' % tar_num)
    fnames = []
    with tarfile.open(tarname, 'w') as tar:
        for line in range(num_lines):
            base_line = [666 for _ in range(2049)]
            base_line[0] = source_num
            base_line[1] = tar_num
            base_line[2] = line
            this_file = fname(line)
            with open(this_file, 'w') as f:
                f.write(json.dumps(base_line))
            tar.add(this_file)
            fnames.append(this_file)
    
        
    for f in fnames:
        try:
            os.unlink(f)
        except:
            pass

        
def make_source(source_num, size_per_tar, total_size):
    num_tars = total_size // size_per_tar
    if total_size % size_per_tar != 0:
        num_tars += 1
    
    num_remaining = total_size    
    for tar_num in range(num_tars):
        this_tar = min(num_remaining, size_per_tar)		        
        make_tar(tar_num, this_tar, source_num=source_num, dir_name='source_id_%02d' % source_num)
        num_remaining -= this_tar    


if __name__ == '__main__':
	for i in range(2):
		make_source(i, 100, 666)