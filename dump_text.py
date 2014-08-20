import argparse, json, os

def dump_text(input_file, out_folder):
    data = json.load(open(input_file))
    data = data["rows"]


    for i in range(len(data)):
        thing = data[i]
        path = os.path.join(out_folder, str(thing["doc"]["_id"]))
        text = thing_to_str(thing["doc"])
        if text:
          with open(path, 'w') as f:
              f.write(text)


def thing_to_str(thing):
    import pdb
    parts = [thing.get('content')]

    for comment in thing.get('comments', []):
        parts.append(comment.get('content'))
        
    #if (len(list(filter(None, parts))) < 1):
      #pdb.set_trace()

    return '\n'.join(list(filter(None, parts))).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('The dumper thing')
    parser.add_argument('input_file', help="file name")
    parser.add_argument('output_folder', help='folder name')

    args = parser.parse_args()

    dump_text(args.input_file, args.output_folder)