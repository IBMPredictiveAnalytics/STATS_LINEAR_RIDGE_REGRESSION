REM ************************************************************************
REM IBM Confidential
REM
REM OCO Source Materials
REM
REM IBM SPSS Products: <Analytic Components>
REM
REM (C) Copyright IBM Corp. 2009, 2010
REM
REM The source code for this program is not published or otherwise divested of its trade secrets,
REM irrespective of what has been deposited with the U.S. Copyright Office.
REM ************************************************************************

REM ************************************************************************
REM IBM Confidential
REM
REM OCO Source Materials
REM
REM IBM SPSS Products: <Analytic Components>
REM
REM (C) Copyright IBM Corp. 2009, 2010
REM
REM The source code for this program is not published or otherwise divested of its trade secrets,
REM irrespective of what has been deposited with the U.S. Copyright Office.
REM ************************************************************************

REM ************************************************************************
REM IBM Confidential
REM
REM OCO Source Materials
REM
REM IBM SPSS Products: <Analytic Components>
REM
REM (C) Copyright IBM Corp. 2009, 2011
REM
REM The source code for this program is not published or otherwise divested of its trade secrets,
REM irrespective of what has been deposited with the U.S. Copyright Office.
REM ************************************************************************

REM ************************************************************************
REM IBM Confidential
REM
REM OCO Source Materials
REM
REM IBM SPSS Products: <Analytic Components>
REM
REM (C) Copyright IBM Corp. 2009, 2022
REM
REM The source code for this program is not published or otherwise divested of its trade secrets,
REM irrespective of what has been deposited with the U.S. Copyright Office.
REM ************************************************************************

@ECHO OFF
IF "%JAVA_HOME%"=="" GOTO SETIT

REM This code assumes that class files already exist in a 'classes' sub-directory.

:NEXT
cd classes
"%JAVA_HOME%\bin\jar" cvf ..\..\alpha_values_peer.jar *
REM jar cvf ../../alpha_values_peer.jar *
cd ..
GOTO END

:SETIT
SET JAVA_HOME="D:\Java\jdk1.8.0_162"
REM /Library/Java/JavaVirtualMachines/jdk1.8.0_311.jdk/Contents/Home
GOTO NEXT

:END
ECHO.
ECHO alpha_values_peer.jar located in parent folder.
pause
