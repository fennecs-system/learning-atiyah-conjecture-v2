from utils import decode, compute_max_dot
# load every data_generation-*.txt file in argument directory

# apply the decode function from makemore.py to each line
# decoded = map(decode, lines)
# gives a list of v, p, k
# compute the dots, k_eval = compute_max_dot(v, p)
# look at dots[k_eval] for each sample
# plot a historgram of these values for each generation all in the same plot
