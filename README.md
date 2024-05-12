# MAGIC_Gamma_Telescope


##### <<For data analysis we create a historgram/print standard deviation/print data count>> ####
 for label in cols[:-1]:
    plt.figure()
    plt.hist(df[df["class"]==1][label], color='blue', label='gamma', alpha=0.7, density=True)
    plt.hist(df[df["class"]==0][label], color='red', label='hadron', alpha=0.7, density=True)
    plt.title=label
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.savefig(f"{label}.png")
plt.show()
std_devs = df.groupby('class').std()
print(std_devs)
class_counts = df['class'].value_counts()
print(class_counts)