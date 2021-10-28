# Merge-Sort-Multiple-Iterations


This is the parallel CUDA implementation of merge sort.

Please view the **"Merge Sort.pptx" file for detailed description and explanation** of project.


# CORE ALGORITHM:
    1)Split the input array into sections of specific size      
    2)Parallelly sort each individual section using a single block of threads                
    3)Iteratively merge sorted sections in order to produce bigger sections, until finally entire array is sorted                        
        All merges in a stage are parallelly done by multiple blocks of threads (Each block responsible for one merge)      
        
        
        
### References:   
    1)CUDA Zone | NVIDIA Developer (https://developer.nvidia.com/cuda-zone)   
    2)Udacity CS344: Intro to Parallel Programming (http://www.udacity.com/wiki/CS344)
    3)Bozidar, Darko & Dobravec, Tomaž. (2015). Comparison of parallel sorting algorithms. 
