import string
import re
import glob

all_letters = string.ascii_letters+"'" + " " + "-" +string.digits

def get_data(orig, dest, all_letters):
	with open(dest, 'w', encoding='UTF-8') as write_file:
		with open(orig, 'r', newline='', encoding='UTF-8') as f:
			spacechecker=0
			text=''
			while True:
				c = f.read(1)
				if not c:
					break
				if c==" " and spacechecker==1:
					continue
				elif c not in all_letters:
					continue
				else:
					if c==" ":
						spacechecker=1
					else:
						spacechecker=0
					write_file.write(c)

with open("freq_data/old_books.txt", "w", encoding='UTF-8') as write_file:
	for filename in glob.glob('freq_output/*.xml'):
		with open(filename, "r", newline='', encoding='UTF-8') as read_file:
			first = True
			for line in read_file:
				if re.match(".*lemma=.*", line):
					word = line.split("</w>")[0].split(">")[1]
					if first == True:
						write_file.write(word)
						first = False
					else:
						write_file.write(" " + word)
				elif re.match(".*</pc>", line):
					word = line.split("</pc>")[0].split(">")[1]
					write_file.write(word)

get_data('freq_data/old_books.txt', 'freq_data/letters_with_dots.txt', all_letters + '●')
