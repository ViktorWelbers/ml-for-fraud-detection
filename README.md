# Coding Challenge - Data Scientist

## Introduction

Online marketplaces are a common target for fraudulent activity where malicious users create fake accounts and upload fake listings seeking monetary gain. In our company, the Customer Service (CS) team manually reviews every listing and assigns ratings (FRAUD or OK), according to their discretion and/or consumer feedback. 
It's very time consuming to manually analyze each listing, and there must be more efficient ways to detect fraud automatically. Since a few months CS started storing all the data for better analysis and understanding of the problem.

## Assignment

Given the dataset provided by CS, your main task is to propose a machine learning solution to support the CS team to remove fraud from the marketplace more efficiently. Your proposal will be presented to both technical and non-technical members of the team. They expect a presentation where they can all understand your approach, the steps taken, your choices, how your proposal can be applied, how well it performs, and its potential benefits for the business. 


## Dataset

An analyst was able to export the data for you, and she was only able to provide this understanding of the columns: 

- modificationTime (listing modification time , e.g 2021-11-11 05:55:19.854)
- siteId (listing country, e.g GERMANY)
- makeId (car makeId, e.g. 23600 makeId corresponds to VW ...)
- firstRegistrationDate (car first registration date, e.g {'month': 7, 'year': 2009})
- damageUnrepaired (whether or not the car is damaged , e.g True)
- priceAmount (car price in Euro, e.g 10000)
- adCreationDate (listing creation date, e.g 2021-11-10 22:21:40)
- date_1 (another date field, e.g 2013-04-26 00:45:12)
- contact_info (seller contact info, e.g)
- adlog_features (a complementary list of listings attributes, e.g seat, mileage, doors, etc)
- rating (final rating to predict, e.g FRAUD)      
 

## Instructions 

Please include both your code sample (e.g Jupyter Notebook) and a small presentation to fulfill the assignment. 
