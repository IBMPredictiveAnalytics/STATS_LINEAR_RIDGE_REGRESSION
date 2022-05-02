/************************************************************************
 ** IBM Confidential
 **
 ** OCO Source Materials
 **
 ** IBM SPSS Products: <Analytic Components>
 **
 ** (C) Copyright IBM Corp. 2009, 2022
 **
 ** The source code for this program is not published or otherwise divested of its trade secrets,
 ** irrespective of what has been deposited with the U.S. Copyright Office.
 ************************************************************************/

package com.ibm.ui.panels;

import com.pasw.framework.common.core.CFEnum;
import com.pasw.framework.common.core.Properties;
import com.pasw.framework.common.session.Session;
import com.pasw.framework.ui.swing.ExtensionObjectSwingUI;
import com.pasw.framework.ui.swing.ManagedPanelContext;
import com.pasw.framework.ui.swing.spi.ManagedPanel;
import com.pasw.framework.ui.swing.spi.ManagedUIElement;
import com.pasw.framework.ui.common.UISession;
import com.pasw.ui.common.control.Control;
import com.pasw.ui.common.control.ControlEvent;
import com.pasw.ui.common.control.ControlListener;
import com.pasw.ui.common.control.ControlManager;
import com.pasw.ui.swing.UIUtilities;
import com.pasw.ui.swing.common.AlertOptionPane;
import com.pasw.ui.swing.common.SwingFeatureUI;
import com.spss.java_client.ui.cf_integration.instance.CFSession;
import com.spss.uitools.res.UIToolResUtil;

import javax.swing.*;
import java.math.BigDecimal;
import java.text.DecimalFormatSymbols;
import java.util.*;

/**
 * Title:       ManagedAlphaValuesPanel
 * Description: A managed panel that contains Mode combo, and Two sets of specify checkboxes,singleValues, start, end, and by numeric fields,
 *
 * Various validation includes:
 * A. For alpha single values:
 *  1. make sure one or more numeric values separated by space, and no duplication within the values
 *  2. When Mode=Fit, only one value is permitted.
 *  3. When Specify check is enabled and checked, no duplication
 * B. For Specify checkboxes: Either individual or grid must be selected
 * C. For Grid: Start/End/By values:
 *  1. make sure no duplication in Single values against grid specification
 *  2. End value should be greater than start value
 * D. For Mode combox: when selection is changed, make sure associated validation stated above is performed accordingly
 *
 * <p>
 * Copyright:   Copyright (c) 2022 IBM Inc. All Rights Reserved
 * Revisions:   Mar 9, 2022 - cbrowning - Initial version
 *              Mar 18, 2022 - yle - implement Alpha related validation work
 */

public class ManagedAlphaValuesPanel implements ManagedPanel, ControlListener {
    public ManagedAlphaValuesPanel() {
        context = null;
        extensionSwingUI = null;
    }

    public void setExtensionSwingUI(ExtensionObjectSwingUI uiObject) {
        this.extensionSwingUI = (SwingFeatureUI) uiObject;
        ControlManager controlManager = this.extensionSwingUI.getControlManager();
        // Setup the ControlListeners
        Control modeControl = controlManager.getControl(MODE);
        if (modeControl != null) {
            modeControl.addControlListener(this);
        }
        Control specifySingleControl = controlManager.getControl(SPECIFY_SINGLE);
        if (specifySingleControl != null) {
            specifySingleControl.addControlListener(this);
        }
        Control specifyGridControl = controlManager.getControl(SPECIFY_GRID);
        if (specifyGridControl != null) {
            specifyGridControl.addControlListener(this);
        }
        Control singleValuesControl = controlManager.getControl(SINGLE_VALUES);
        if (singleValuesControl != null) {
            singleValuesControl.addControlListener(this);
        }
        Control startControl = controlManager.getControl(START);
        if (startControl != null) {
            startControl.addControlListener(this);
        }
        Control endControl = controlManager.getControl(END);
        if (endControl != null) {
            endControl.addControlListener(this);
        }
        Control byControl = controlManager.getControl(BY);
        if (byControl != null) {
            byControl.addControlListener(this);
        }
        Control metricControl = controlManager.getControl(METRIC);
        if (metricControl != null) {
            metricControl.addControlListener(this);
        }
    }

    // ------------------------------- ManagedPanel Implementation ------------------------------- //

    /**
     * This is the first method to be called on the manager after
     * it has been created. This should be used to initialise member variables. Because the
     * UI is still being created, objects exposed via the context object will not be fully
     * initialised. If possible, implementations should defer calling methods on the context
     * until other SPI methods are invoked.
     *
     * @param panelId    the panel id of this panel if one was specified
     * @param properties a collection of string properties associated with the panel declaration
     * @param context    the managed panel context
     */
    public void initManagedPanel(String panelId, Properties properties, ManagedPanelContext context) {
        this.context = context;
        SwingUtilities.invokeLater(() -> setExtensionSwingUI(context.getExtensionObjectUI()));
    }

    /**
     * Creates a new ManagedUIElement for a custom ChildElement.
     *
     * @param id The value of the "id" attribute in the extension XML
     * @return A newly created ManagedUIElement or null
     */
    public ManagedUIElement createManagedUIElement(String id) {
        return null;
    }

    /**
     * Called to notify the managed panel that the window which contains the managed panel is
     * being destroyed. This provides an opportunity to free
     * any resources than need to be freed explicitly. This is the
     * last method called on this object.
     */
    public void disposeManagedPanel() {
        ControlManager controlManager = this.extensionSwingUI.getControlManager();

        // Remove the ControlListeners
        Control specifySingleControl = controlManager.getControl(SPECIFY_SINGLE);
        if (specifySingleControl != null) {
            specifySingleControl.removeControlListener(this);
        }
        Control specifyGridControl = controlManager.getControl(SPECIFY_GRID);
        if (specifyGridControl != null) {
            specifyGridControl.removeControlListener(this);
        }
        Control singleValuesControl = controlManager.getControl(SINGLE_VALUES);
        if (singleValuesControl != null) {
            singleValuesControl.removeControlListener(this);
        }
        Control startControl = controlManager.getControl(START);
        if (startControl != null) {
            startControl.removeControlListener(this);
        }
        Control endControl = controlManager.getControl(END);
        if (endControl != null) {
            endControl.removeControlListener(this);
        }
        Control byControl = controlManager.getControl(BY);
        if (byControl != null) {
            byControl.removeControlListener(this);
        }
        Control metricControl = controlManager.getControl(METRIC);
        if (metricControl != null) {
            metricControl.removeControlListener(this);
        }
    }

    // ----------------------------- ControlListener Implementation ------------------------------- //

    /**
     * Listen when control value is changed
     */
    public void controlValueChanged(ControlEvent event) {
        // When loadingState is true, we just ignore and return, fix https://github.ibm.com/SPSS/stats_java_ui/issues/5684
        Session session = extensionSwingUI.getFeature().getSession();
        if (session instanceof CFSession && ((CFSession)session).getLoadingState())
            return;
        String propertyName = event.getPropertyName();
        if (propertyName.equals(SPECIFY_SINGLE) || propertyName.equals(SPECIFY_GRID)) {
            validateCheckboxPair(propertyName);
        } else if (propertyName.equals(SINGLE_VALUES) || propertyName.equals(START) || propertyName.equals(END) || propertyName.equals(BY)) {
            checkValidNumbers(propertyName);
        } else if (propertyName.equals(MODE)) {
            validateCheckboxPair(SPECIFY_SINGLE);
            checkOneAlphaValue();
        } else if (propertyName.equals(METRIC)) {
            checkValidNumbers(START);
            checkValidNumbers(END);
            checkValidNumbers(SINGLE_VALUES);
        }
    }

    public void controlSelectionChanged(ControlEvent event) {
    }

    // -------------------------------------------------------------------------------------------- //
    private void validateCheckboxPair(String propertyName) {
        Boolean specifySingleChecked = (Boolean) extensionSwingUI.getControlValue(SPECIFY_SINGLE);
        Control specifyGridControl = extensionSwingUI.getControlManager().getControl(SPECIFY_GRID);
        Boolean specifyGridChecked = (Boolean) extensionSwingUI.getControlValue(SPECIFY_GRID);
        if (!specifySingleChecked && (!specifyGridControl.isEnabled() || !specifyGridChecked)) {
            AlertOptionPane.showErrorMessageDialog(extensionSwingUI.getRootComponent(),
                    extensionSwingUI.getSwingResourceProvider().getString("single_grid_alpha_select_error.MANAGED"),
                    getUISession().getApplication().getApplicationBranding().getApplicationName());
            Control ctrl = extensionSwingUI.getControlManager().getControl(propertyName);
            String targetCtrlName = propertyName;
            if (propertyName.equals(SPECIFY_SINGLE) && specifyGridControl.isEnabled())
                targetCtrlName = SPECIFY_GRID;
            else if (propertyName.equals(SPECIFY_GRID))
                targetCtrlName = SPECIFY_SINGLE;
            // Remove listener before reset the value
            ctrl.removeControlListener(this);
            extensionSwingUI.getControlManager().getControl(targetCtrlName).setControlValue(targetCtrlName, true);
            ctrl.addControlListener(this);
        }
        // Need to re-check duplicates here because checkbox state change causes textbox enable/disable to change
        handleDuplicates(SINGLE_VALUES);
    }

    /**
     * Check to validate the field which was just clicked away
     *
     * @param propertyName The current propertyName to be clicked away (focus lost)
     */
    private void checkValidNumbers(String propertyName) {
        // check if current value is all numeric, or positive when required
        checkCurrentValue(propertyName);

        Double startValue = (Double) extensionSwingUI.getControlValue(START);
        Double endValue = (Double) extensionSwingUI.getControlValue(END);
        Double byValue = (Double) extensionSwingUI.getControlValue(BY);
        // Check Start and End value to validate
        if (propertyName.equals(START) || propertyName.equals(END) || propertyName.equals(BY)) {
            if (startValue != null && endValue != null && byValue != null && endValue <= startValue) {
                AlertOptionPane.showErrorMessageDialog(extensionSwingUI.getRootComponent(),
                        extensionSwingUI.getSwingResourceProvider().getString("end_less_than_start_error.MANAGED"),
                        getUISession().getApplication().getApplicationBranding().getApplicationName());
                // Switch the start and end value
                ControlManager cm = extensionSwingUI.getControlManager();
                Control startCtrl = cm.getControl(START);
                Control endCtrl = cm.getControl(END);
                startCtrl.removeControlListener(this);
                endCtrl.removeControlListener(this);
                cm.getControl(START).setControlValue(START, endValue);
                cm.getControl(END).setControlValue(END, startValue);
                startCtrl.addControlListener(this);
                endCtrl.addControlListener(this);
                UIUtilities.getInstance().requestFocusForControl(cm.getControl(START));
            }
        }
        if (propertyName.equals(SINGLE_VALUES))
            checkOneAlphaValue();

        // Handle duplicates
        handleDuplicates(propertyName);
    }

    /**
     * Check SingleValues field to make sure that when Mode=Fit, only one value is permitted
     */
    private void checkOneAlphaValue() {
        CFEnum m = (CFEnum) extensionSwingUI.getControlValue(MODE);
        Object singleValue = extensionSwingUI.getControlValue(SINGLE_VALUES);
        String[] valArray = singleValue.toString().replaceAll("(^\\s+|\\s+$)", "").split("\\s+");
        // When Mode=Fit: Only one value of alpha is allowed
        if (m.toString().equals(MODE_FIT)) {
            if (valArray.length > 1) {
                showErrorAndUpdate(SINGLE_VALUES, valArray[0], "only_one_alpha_value_error.MANAGED");
            }
        }
    }

    /**
     * Check for duplicate values - between singleValues and values specified in grid
     */
    private boolean checkDuplicate(String propertyName, Double[] doubleValues, Double startValue, Double endValue,
                                   Double byValue, Set<Double> resultSet) {
        boolean hasDuplicate;

        Set<Double> set = new HashSet<>();
        if (propertyName.equals(SINGLE_VALUES)) {
            // check duplicate WITHIN single values
            List<Double> list = Arrays.asList(doubleValues);
            boolean hasDuplicate1 = false;
            for (Double l : list) {
                if(!set.add(l)) {
                    hasDuplicate1 = true;
                } else {
                    resultSet.add(l);
                }
            }

            ControlManager cm = extensionSwingUI.getControlManager();
            Control startControl = cm.getControl(START);
            // Check duplication comparing with grid values
            boolean hasDuplicate2 = false;
            if (startControl.isEnabled() && startValue != null && endValue != null && byValue != null)
                hasDuplicate2 = hasDuplicateAgainstGrid(doubleValues, startValue, endValue, byValue, resultSet);

            hasDuplicate = hasDuplicate1 | hasDuplicate2;
        } else {
            resultSet.addAll(Arrays.asList(doubleValues));
            hasDuplicate = hasDuplicateAgainstGrid(doubleValues, startValue, endValue, byValue, resultSet);
        }
        return hasDuplicate;
    }

    /**
     * Check duplication between singleValues against all values defined by grid
     * @return true if duplication is found, false otherwise
     */
    private boolean hasDuplicateAgainstGrid(Double[] doubleValues, Double startValue, Double endValue, Double byValue, Set<Double> resultSet) {
        boolean hasDuplicate = false;
        if (startValue != null && endValue != null && byValue != null) {
            ArrayList<Double> all = getAllParamValuesInGrid(startValue, endValue, byValue);
            // Further remove the duplicates against grid
            for (Double v : doubleValues) {
                if (all.contains(v)) {
                    hasDuplicate = true;
                    resultSet.remove(v);
                }
            }
        }
        return hasDuplicate;
    }

    private void handleDuplicates(String propertyName) {
        Object singleValue = extensionSwingUI.getControlValue(SINGLE_VALUES);
        Control singleCtrl = extensionSwingUI.getControlManager().getControl(SINGLE_VALUES);
        Double startValue = (Double) extensionSwingUI.getControlValue(START);
        Double endValue = (Double) extensionSwingUI.getControlValue(END);
        Double byValue = (Double) extensionSwingUI.getControlValue(BY);
        if (!singleCtrl.isEnabled() || singleValue == null || singleValue.toString().length() == 0 )
            return;

        Double[] doubleValues = null;
        String[] valArray = null;
        if (singleCtrl.isEnabled()) {
            // ignore all trailing or heading white spaces
            valArray = singleValue.toString().replaceAll("(^\\s+|\\s+$)", "").split("\\s+");
            doubleValues = Arrays.stream(valArray).map(Double::valueOf).toArray(Double[]::new);
        }

        Set<Double> resultSet = new HashSet<>();
        Set<String> temp = new HashSet<>();
        StringBuilder builder = new StringBuilder();
        if (doubleValues != null && checkDuplicate(propertyName, doubleValues, startValue, endValue, byValue, resultSet)) {
            // remove duplicates from user input to maintain the original input
            for (String v : valArray) {
                if (resultSet.contains(Double.valueOf(v))) {
                    if (temp.add(v)) {
                        builder.append(v);
                        builder.append(" ");
                    }
                }
            }
            // Duplicate values have been removed
            String newValues = builder.toString().trim();
            showErrorAndUpdate(SINGLE_VALUES, newValues, "alpha_duplicate_value_error.MANAGED");
        }
    }

    /**
     * Check current control value to see if the value is empty or numeric values
     * @param propertyName the current propertyName in concern
     */
    private void checkCurrentValue(String propertyName) {
        Object val = extensionSwingUI.getControlValue(propertyName);

        // Note: For number field, we would get either numeric value or null, won't get empty string
        // Note: Empty field checking resides in extension.xml, we will ignore here
        if (val == null || val.toString().trim().length() == 0)    return;

        CFEnum metricVal = (CFEnum) extensionSwingUI.getControlValue(METRIC);
        boolean isLinear = metricVal.toString().equals(LINEAR);
        // For number field, we don't need to check here because the CDB ui already guards it
        if (propertyName.equals(SINGLE_VALUES) && val instanceof String) {
            char decimalSep = new DecimalFormatSymbols(UIToolResUtil.getSPSSLocale()).getDecimalSeparator();
            String v = val.toString().trim();
            StringBuilder builder = new StringBuilder();
            // Check if all numbers, or decimal separator, or space for singleValues field
            boolean hasInvalid = false;
            for (int i = 0; i < v.length(); i++) {
                char ch = v.charAt(i);
                if (Character.isDigit(ch) || ch == decimalSep || ch == ' ' || (!isLinear && ch == '-' ) ){
                    builder.append(ch);
                } else {
                    hasInvalid = true;
                }
            }
            if (hasInvalid) {
                String msg = isLinear ? "alpha_numeric_positive_value_error.MANAGED" : "alpha_numeric_value_error.MANAGED";
                showErrorAndUpdate(propertyName, builder.toString(), msg);
            }
            checkOneAlphaValue();
        } else if (isLinear && (propertyName.equals(START) || propertyName.equals(END))) {
            // For Linear: Check if the value should be positive only
            Double value = (Double) extensionSwingUI.getControlValue(propertyName);
            if (value < 0)
                // do not return false to follow through additional checking
                showErrorAndUpdate(propertyName, Math.abs(value), "alpha_positive_value_error.MANAGED");
        }
    }

    /**
     * Get all the param values defined by grid
     * @param start The start value
     * @param end   The end value
     * @param by    The by value
     * @return the ArrayList of Double containing all those values
     */
    private ArrayList<Double> getAllParamValuesInGrid(Double start, Double end, Double by) {
        ArrayList<Double> list = new ArrayList<>();

        for (int i = 0; i <= 50; i++) {
            // Using BigDecimal to fix issue: https://github.ibm.com/SPSS/stats_defects/issues/2038
            // Because the drawback of primitive type float or double number's calculation resulting
            // unexpected number sometimes, for example, 0.1*3 being 0.30000000000000004 instead of just 0.3,
            // making checking for duplication fails, Use BigDecimal solves this issue
            BigDecimal bs = new BigDecimal(String.valueOf(start));
            BigDecimal bb = new BigDecimal(String.valueOf(by));
            BigDecimal bi = new BigDecimal(String.valueOf(i));
            double result = bi.multiply(bb).add(bs).doubleValue();
            if (result <= end) {
                list.add(result);
            } else
                break;
        }
        return list;
    }

    /**
     * Obtain the UISession
     *
     * @return the UISession
     */
    private UISession getUISession() {
        return extensionSwingUI.getSwingResourceProvider().getUISession();
    }

    /**
     * The Common method to show error alert and update with valid values, and move focus to the control
     * @param propertyName The current propertyName in concern
     * @param newVal    The new valid value Object to set, could be Double, String, Integer etc.
     * @param errorKey The error message id
     */
    private void showErrorAndUpdate(String propertyName, Object newVal, String errorKey) {
        // Show error message
        AlertOptionPane.showErrorMessageDialog(extensionSwingUI.getRootComponent(),
                extensionSwingUI.getSwingResourceProvider().getString(errorKey),
                getUISession().getApplication().getApplicationBranding().getApplicationName());
        ControlManager cm = extensionSwingUI.getControlManager();
        Control currentCtrl = cm.getControl(propertyName);
        currentCtrl.removeControlListener(this);
        cm.getControl(propertyName).setControlValue(propertyName, newVal);
        currentCtrl.addControlListener(this);
        UIUtilities.getInstance().requestFocusForControl(cm.getControl(propertyName));
    }

    // IDs of controls
    private final static String MODE = "mode";
    private final static String SPECIFY_SINGLE = "singleCheck";
    private final static String SPECIFY_GRID = "gridCheck";
    private final static String SINGLE_VALUES = "singleValues";

    private final static String START = "start";
    private final static String END = "end";
    private final static String BY = "by";

    private final static String METRIC = "metric";

    // Mode combo selection value - first
    private final static String MODE_FIT = "mode1";
    // Metric radio group selection - first radio selection
    private final static String LINEAR = "linearRadio";

    /**
     * @noinspection FieldCanBeLocal, UnusedDeclaration
     */
    private ManagedPanelContext context;

    /**
     * The main ExtensionObjectSwingUI for this feature.
     */
    private SwingFeatureUI extensionSwingUI;
}

