import matplotlib.pyplot as plt

bleu_1 = [61.92, 63.09, 64.92, 50.66, 48.4, 55.27, 62.7, 61.87]
bleu_2 = [43.33, 44.31, 46.82, 26.37, 28.78, 30.18, 44.12, 43.52]
bleu_3 = [29.49, 30.32, 32.8, 15.24, 17.22, 17.69, 30.19, 29.78]
bleu_4 = [19.67, 20.15, 22.1, 9.57, 10.37, 11.48, 20.08, 19.88]
rouge_l = [42.72, 43.51, 44.73, 36.54, 33.46, 37.7, 43.45, 43.11]
cider = [42.86, 42.64, 46.56, 7.89, 13.34, 9.54, 39.17, 40.69]
spice = [11.61, 11.49, 12.23, 4.29, 5.41, 4.3, 11.18, 11.09]
meteor = [18.04, 18.27, 19.02, 11.43, 11.87, 11.57, 18, 17.94]
models = ["Baseline (BS)", "BS + 1 Enc Rem", "BS + 1 Enc Rem + 1 Dec Rem", "BS + Input Size 288 (Resize)", "BS + Input Size 288 (Center Crop)", "BS + Input Size 192 (Resize)", "BS + 1 Enc Rem + Input Size 288 (Resize)", "BS + 1 Enc Rem + 1 Dec Rem + Input Size 288 (Resize)"]
parameter_counts = [233803076, 229372740, 224627524, 233799044, 233799044, 233796164, 229368708, 224623492]
flops = [158005560832, 120903841280, 115054609920, 123642488320, 123642488320, 123615946240, 120866682368, 113451099648]
latencies = []

def plot(models, parameter_counts, flops, latencies):
  _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
  ax1.plot(parameter_counts, rouge_l, '--o', color="red")
  ax1.set_title('Parameter Counts')
#   ax1.get_xaxis().get_major_formatter().set_useOffset(False)
#   ax1.get_xaxis().get_major_formatter().set_scientific(False)
#   ax1.set_xticklabels(ax1.get_xticks(), rotation=90)
  ax1.grid(True, linestyle='-', alpha=0.8, color='white')
  ax1.set_facecolor('lightgrey')
  ax1.set_ylabel("ROUGE-L")

  ax2.plot(latencies, rouge_l, '--o', color="red")
  ax2.set_title('Latencies')
  ax2.set_facecolor('lightgrey')
  ax2.grid(True, linestyle='-', alpha=0.8, color='white')

  ax3.plot(flops, rouge_l, '--o', color="red")
  ax3.set_title('FLOPs')
  ax3.set_facecolor('lightgrey')
  ax3.grid(True, linestyle='-', alpha=0.8, color='white')

#   fig.text(0.0, 0.6, 'ROUGE-L', va='center', rotation='vertical')
  for i, model in enumerate(models):
      ax1.text(parameter_counts[i], rouge_l[i], model, fontsize=10, va='bottom')
      ax2.text(latencies[i], rouge_l[i], model, fontsize=10, va='bottom')
      ax3.text(flops[i], rouge_l[i], model, fontsize=10, va='bottom')

  # Adjust spacing between subplots
  return plt

# Varying the input size
models = ["BS + Input Size 192 (Resize)", "BS + Input Size 288 (Resize)", "Baseline (BS)"]
parameter_counts = [233796164, 233799044, 233803076]
flops = [123615946240, 123642488320, 158005560832]
latencies = [0, 1, 2]
rouge_l = [37.7, 36.54, 42.72]

plt1 = plot(models, parameter_counts, flops, latencies)
plt1.savefig("./benchmarking/plots/q7i.jpg")
# plt1.show()

models = ["BS + Input Size 288 (Resize)", "BS + Input Size 288 (Center Crop)"]
parameter_counts = [233799044, 233799044]
flops = [123642488320, 123642488320]
latencies = [0, 1]
rouge_l = [36.54, 33.46]
plt1 = plot(models, parameter_counts, flops, latencies)
plt1.savefig("./benchmarking/plots/q7ii.jpg")

# Varying number of parameters
models = ["BS + 1 Enc Rem + 1 Dec Rem", "BS + 1 Enc Rem", "Baseline (BS)"]
parameter_counts = [224627524, 229372740, 233803076]
flops = [115054609920, 120903841280, 158005560832]
latencies = [0, 1, 2]
rouge_l = [44.73, 43.51, 42.72]

plt1 = plot(models, parameter_counts, flops, latencies)
plt1.savefig("./benchmarking/plots/q9.jpg")


# Putting it all together
models = ["BS + 1 Enc Rem + 1 Dec Rem + Input Size 288 (Resize)", "BS + 1 Enc Rem + Input Size 288 (Resize)", "Baseline (BS)"]
parameter_counts = [224623492, 229368708, 233803076]
flops = [113451099648, 120866682368, 158005560832]
latencies = [0, 1, 2]
rouge_l = [43.11, 43.45, 42.72]

plt1 = plot(models, parameter_counts, flops, latencies)
plt1.savefig("./benchmarking/plots/q11.jpg")