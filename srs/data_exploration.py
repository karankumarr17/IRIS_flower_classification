import seaborn as sns
import matplotlib.pyplot as plt

def explore_data(data):
    """Display dataset overview and visualizations."""
    print("\n📊 Dataset Overview:")
    print(data.head())

    print("\nℹ️ Dataset Info:")
    print(data.info())

    print("\n📈 Dataset Statistics:")
    print(data.describe())

    print("\n🎨 Visualizing Pairplot:")
    sns.pairplot(data, hue='species', palette='Set1')
    plt.show()
