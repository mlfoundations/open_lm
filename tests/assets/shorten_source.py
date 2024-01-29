import webdataset as wds
import os

def main():

    indir = "./source_1/"
    outdir = "./source_seq_len_16/"

    for infile in os.listdir(indir):

        if not infile.endswith("tar"):
            continue

        outfile = os.path.join(outdir, infile)

        ds = wds.WebDataset(os.path.join(indir, infile)).decode()

        with open(outfile, "wb") as f:
            with wds.TarWriter(f) as writer:
                for i, x in enumerate(ds):
                    if i >= 100:
                        break
                    new_item = {}
                    new_item["__key__"] = x["__key__"]
                    new_item["__url__"] = x["__url__"]
                    new_item["json"] = []
                    for i in range(16):
                        item = x["json"][i]
                        if isinstance(item, int):
                            new_item["json"].append(x["json"][i] % 10)
                        else:
                            new_item["json"].append(x["json"][i])
                    new_item["json"].append([x["json"][-1]])
                    print(x.keys())
                    print(type(x["json"]))
                    print(len(x["json"]))
                    print(len(new_item["json"]))
                    writer.write(new_item)


if __name__ == "__main__":
    main()