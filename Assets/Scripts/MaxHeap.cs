using System.Collections;
using System.Collections.Generic;


public class MaxHeap<T>
{
    private List<T> priorityQueue = new List<T>();
    private int getElementValue;
    private int numberOfElements;

    public MaxHeap(int maxSize, int getElementValue)
    {
        //this.priorityQueue = new List(maxSize);
        this.numberOfElements = -1;
        this.getElementValue = getElementValue;
    }

    private int half(double k)
    {
        return (int)System.Math.Floor(k / 2);
    }


    
}
