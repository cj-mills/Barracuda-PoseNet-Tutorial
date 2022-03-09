using System.Collections.Generic;
using UnityEngine;
using System;

#if UNITY_EDITOR
using UnityEditor;
/// <summary>
/// A custom property drawer that displays a list variable as a popup selection field
/// </summary>
[CustomPropertyDrawer(typeof(ListToPopupAttribute))]
public class ListToPopupDrawer : PropertyDrawer
{
    // Set default list index to 0
    public int selectedIndex = 0;

    /// <summary>
    /// Draw custom property drawer
    /// </summary>
    /// <param name="position"></param>
    /// <param name="property"></param>
    /// <param name="label"></param>
    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
    {
        // Cast property attribute as a ListToPopupAttribute
        ListToPopupAttribute atb = attribute as ListToPopupAttribute;
        List<string> stringList = null;

        if (atb.myType.GetField(atb.propertyName) != null)
        {
            // Populate stringList with values for the List variable
            stringList = atb.myType.GetField(atb.propertyName).GetValue(atb.myType) as List<string>;
        }

        if (stringList != null && stringList.Count != 0)
        {
            // Create a popup selection field
            selectedIndex = EditorGUI.Popup(position, property.name, selectedIndex, stringList.ToArray());
            // Get the corresponding string value for the selcted index
            property.stringValue = stringList[selectedIndex];
        }
        else
        {
            // Draw a default property field is the stringList is empty
            EditorGUI.PropertyField(position, property, label);
        }
    }
}
#endif

/// <summary>
/// A custom property attribute
/// </summary>
public class ListToPopupAttribute : PropertyAttribute
{
    public Type myType;
    public string propertyName;

    public ListToPopupAttribute(Type _myType, string _propertyName)
    {
        myType = _myType;
        propertyName = _propertyName;
    }
}