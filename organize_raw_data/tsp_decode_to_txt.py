import gzip
import os
import chardet

# after extracting the tar files, i'm decoding the files into /tsp_decoded
def read_gz_file_all():
    source_folder = '../tsp_extracted/'
    destination_folder = '../tsp_decoded/'

    for filename in os.listdir(source_folder):
        if filename.endswith('.tsp.gz'):
            # open file and decode contents
            source_file_path = os.path.join(source_folder, filename)
            with gzip.open(source_file_path, 'rb') as f:
                decompressed_data = f.read()
                encoding = chardet.detect(decompressed_data)['encoding']
                decoded_data = decompressed_data.decode(encoding)

            # make and name of the file you want to insert the decoded data into
            txt_file_name = os.path.splitext(filename)[0]
            destination_path = os.path.join(destination_folder, txt_file_name + '.txt')
           
            # write the decoded data into destination file
            with open(destination_path, 'w') as file:
                file.write(decoded_data)
            



read_gz_file_all()