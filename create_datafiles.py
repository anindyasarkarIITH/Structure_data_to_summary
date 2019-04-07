''' create two vocabulary files for fields and content union summary. create seperate chunked binary files for 2 examples.'''

import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2



dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

#tokenized_examples_dir = "/home/anindya/Documents/structure_data_to_summary/tokenized_examples"
#dm_tokenized_stories_dir = "dm_stories_tokenized"
finished_files_dir = "finished_files"
chunks_dir = os.path.join(finished_files_dir, "chunked")

VOCAB_SIZE = 200000
CHUNK_SIZE = 2 #1000 # num examples per chunk, for the chunked data


def chunk_file(set_name):
  in_file = 'finished_files/%s.bin' % set_name
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1


def chunk_all():
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  #for set_name in ['train', 'val', 'test']:
  for set_name in ['train']:
    print "Splitting %s data into chunks..." % set_name
    chunk_file(set_name)
  print "Saved chunked data in %s" % chunks_dir

def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines


def get_art_abs(story_file):
  lines = read_text_file(story_file)

  # Lowercase everything
  lines = [line.lower() for line in lines]

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  #lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
  summary = []
  fields = []
  contents = []
  next_is_field = False
  next_is_content = False
  for idx,line in enumerate(lines):
    if line == "":
      continue # empty line
    elif line.startswith("@fields") and not line.startswith("@contents"):
      next_is_field = True
      next_is_content = False
    elif next_is_field and not line.startswith("@contents"):
      fields.append(line)
    elif line.startswith("@contents") and not line.startswith("@fields"):
      next_is_field = False
      next_is_content = True
    elif next_is_content and not line.startswith("@fields"):
      contents.append(line)
    else:
      summary.append(line)

  # Make article into a single string
  fields_example = ' '.join(fields)
  contents_example =  ' '.join(contents)

  # Make abstract into a signle string, putting <s> and </s> tags around the sentences
  summary_example = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in summary])

  return summary_example, fields_example, contents_example


def write_to_bin(examples_dir, out_file, makevocab=True):
  """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
  #print "Making bin file for URLs listed in %s..." % url_file
  #url_list = read_text_file(url_file)
  #url_hashes = get_url_hashes(url_list)
  #story_fnames = [s+".story" for s in url_hashes]
  story_fnames = os.listdir(examples_dir)
  num_stories = len(story_fnames)

  if makevocab:
    vocab_counter_content_summary = collections.Counter()
    vocab_counter_field = collections.Counter()

  with open(out_file, 'wb') as writer:
    for idx,s in enumerate(story_fnames):
      if idx % 1000 == 0:
        print "Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories))

      # Look in the tokenized story dirs to find the .story file corresponding to this url
      if os.path.isfile(os.path.join(examples_dir, s)):
        story_file = os.path.join(examples_dir, s)
      else:
        print "Error: Couldn't find tokenized story file %s in either tokenized story directories %s and %s. Was there an error during tokenization?" % (s, examples_dir, dm_tokenized_stories_dir)
        # Check again if tokenized stories directories contain correct number of files
        print "Checking that the tokenized stories directories %s contain correct number of files..." % (examples_dir)
        raise Exception("Tokenized stories directories %s contain correct number of files but story file %s found in neither." % (examples_dir, s))

      # Get the strings to write to .bin file
      summary_example, field_example, content_example = get_art_abs(story_file)
      print (content_example)
      print (field_example)
      print (summary_example)
     
      
      # Write to tf.Example
      tf_example = example_pb2.Example()
      tf_example.features.feature['summary_example'].bytes_list.value.extend([summary_example])
      tf_example.features.feature['content_example'].bytes_list.value.extend([content_example])
      tf_example.features.feature['field_example'].bytes_list.value.extend([field_example])
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

      # Write the vocab to file, if applicable
      if makevocab:
        summ_tokens = summary_example.split(' ')
        cont_tokens = content_example.split(' ')
        field_tokens = field_example.split(' ')
        summ_tokens = [t for t in summ_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
        tokens = summ_tokens + cont_tokens
        tokens = [t.strip() for t in tokens] # strip
        tokens = [t for t in tokens if t!=""] # remove empty
        vocab_counter_content_summary.update(tokens)
        f_tokens = [t.strip() for t in field_tokens] # strip
        f_tokens = [t for t in f_tokens if t!=""] # remove empty
        vocab_counter_field.update(f_tokens)

  print "Finished writing file %s\n" % out_file
  
  # write vocab to file
  if makevocab:
    print "Writing vocab file..."
    with open(os.path.join(finished_files_dir, "vocab_content_summary"), 'w') as writer:
      for word, count in vocab_counter_content_summary.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')
    print "Finished writing vocab file for summary and content"
    with open(os.path.join(finished_files_dir, "vocab_field"), 'w') as writer:
      for word, count in vocab_counter_field.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')
    print "Finished writing vocab file for field"
    

def check_num_stories(stories_dir, num_expected):
  num_stories = len(os.listdir(stories_dir))
  if num_stories != num_expected:
    raise Exception("stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))


if __name__ == '__main__':
  
  examples_dir = "/home/anindya/Documents/structure_data_to_summary/examples"
  
  # Create some new directories
  if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

  # Read the tokenized stories, do a little postprocessing then write to bin files
  write_to_bin(examples_dir, os.path.join(finished_files_dir, "train.bin"))
  
  # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples,
  #and saves them in finished_files/chunks
  chunk_all() 
  
