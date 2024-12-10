
def import_reward_model(args):
    reward_model = None
    if args.rd_method == 'RRD' or args.rd_method == 'RRD_unbiased':
        from reward_model.rrd import RRDRewardDecomposer
        reward_model = RRDRewardDecomposer(args)
    elif args.rd_method == 'VIB':
        from reward_model.VIB import IBRewardDecomposer
        reward_model = IBRewardDecomposer(args)
    elif args.rd_method == 'RD':
        from reward_model.RD import RDRewardDecomposer
        reward_model = RDRewardDecomposer(args)
    elif args.rd_method == 'LaRe_RRD' or args.rd_method == 'LaRe_RRDu':
        from reward_model.LLMrd import LLMRewardDecomposer
        reward_model = LLMRewardDecomposer(args)
    elif args.rd_method == 'LaRe_RD':
        from reward_model.LLMrd import LLMRDRewardDecomposer
        reward_model = LLMRDRewardDecomposer(args)
    elif args.rd_method == 'Diaster':
        from reward_model.Diaster import DiasterRewardDecomposer
        reward_model = DiasterRewardDecomposer(args)
    return reward_model