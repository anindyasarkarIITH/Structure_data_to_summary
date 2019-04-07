''' each example having a seperate field,seperate content and seperate summary from train.box train.nb and train.sent file.'''

import sys
import os

examples_dir = "examples"


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def process_field_content_pair(field_content_pair):
  field_content_list = field_content_pair.split("\t")
  content_example = list() ; field_example = list()
  for field_content in field_content_list:
    field = field_content.split(":")[0]
    content = field_content.split(":")[1]
    content_example.append(content)
    if is_number(field.split("_")[-1]):
      field_example.append(field.split("_")[:-1][0])
    else:
      field_example.append(field)
  return (field_example,content_example)
  


if __name__ == '__main__':

  ## read all the files
  field_content_dir = open("/home/anindya/Documents/structure_data_to_summary/Data/train.box","r")
  summary_dir = open("/home/anindya/Documents/structure_data_to_summary/Data/train.sent","r")
  num_sent_summary = open("/home/anindya/Documents/structure_data_to_summary/Data/train.nb","r")
  
  # create different example file with each seperate field for field,content,summary.and also create tokenized version of it.
  if not os.path.exists(examples_dir): os.makedirs(examples_dir)
  num_sent_summary_list = num_sent_summary.read().splitlines() # list contains number of sentences
  summary_dir_list = summary_dir.read().splitlines() # list contains summary
  count_summary_sent = 0
  for i,field_content_pair in enumerate(field_content_dir.read().splitlines()):
    #print (summary_dir_list[i])
    num_sent_example = num_sent_summary_list[i]
    summary_example = summary_dir_list[count_summary_sent : count_summary_sent + int(num_sent_example)]
    count_summary_sent = count_summary_sent + int(num_sent_example)
    #print (summary_example)
    #print (field_content_pair)
    field_example,content_example = process_field_content_pair(field_content_pair)
    assert len(field_example) == len(content_example)
    print (field_example)
    print (content_example)
    # write to the example file summary_example,field_example,content_example
    fh = open(examples_dir+"/example_"+str(i)+".txt","w")
    for summary in summary_example:
      fh.write(summary)
      fh.write(" ")
    fh.write("\n\n\n@fields\n")
    for field in field_example:
      fh.write(field)
      fh.write(" ")
    fh.write("\n\n\n@contents\n")
    for content in content_example:
      fh.write(content)
      fh.write(" ")
    fh.close()
    
        

    
  
    
