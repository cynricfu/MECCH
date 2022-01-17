join_token = "=>"


def sub_metapaths(metapath_str):
    tokens = metapath_str[3:].split(join_token)
    return ["mp:" + join_token.join(tokens[i:]) for i in range(0, len(tokens) - 2, 2)]


def get_src_ntypes(metapath_str):
    tokens = metapath_str[3:].split(join_token)
    src_ntypes = [tokens[i] for i in range(0, len(tokens) - 2, 2)]
    return list(set(src_ntypes))
