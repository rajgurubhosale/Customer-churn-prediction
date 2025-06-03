# **Buisness overview**

### Buisness Problem:
Telecom comapnies suffers from the Voluntary churn,which occurs when customers leave the service provider<br> for a competitor.
Telecom companies spend hundreds of dollars<br> to acquire a new customer and when that customer leaves, the company not only loses the future revenue <br>from that customer but also the resources spend to acquire that customer. Churn Reduces profitability.<br>
**Study shows that increasing retention rate by 5 % increase profits by 25% to 95%😲.**<br>
To achieve this, it’s crucial to identify the customers who are most likely to leave (churn),<br> so effective retention strategies can be implemented.

### Why Churn is major problem:
* **Loss of Future Revenue**: When a customer leaves, the company loses all future payments from them
*  **Wasted Acquisition Cost**: Telecom companies spend a lot of money to gain a new customer—if they leave, that money is wasted.
* **Direct hit on ROI**, profitability, and brand loyalty.
*  **Wrong Targeting Wastes Money**: Offering discounts to customers who weren’t going to leave leads to unnecessary costs.

### 🎯 Business Goal 
The objective is to develop a **data-driven churn prediction model** that accurately flags customers at high risk of leaving
his enables the business to allocate retention resources more efficiently—focusing efforts and budget only on those likely to churn, rather than applying broad, untargeted strategies. Additionally, the EDA and the model will help uncover key behavioral and service-related drivers behind churn, guiding long-term improvements.

### Telecom companies use two main approaches to tackle this problem:

1)Untargeted Approach:
Focuses on improving the product and using mass advertising to increase brand loyalty.<br>
(mass advertising can be costly without targeting the right customers. It also lacks personalization, reducing its effectiveness in retaining customers.)

2)Targeted Approach:
Identifies customers likely to churn and offers interventions to keep them.


### Role of Predictive Modeling:
In the targeted approach, companies try to predict which customers are likely to churn and offer them special programs or incentives. However, if the churn predictions are inaccurate, companies may waste money on customers who would have stayed. There are various predictive modeling techniques to forecast churn, such as neural networks, logistic regression, and survival analysis. 
logistic regression is ideal for binary outcomes (churn/no churn), while neural networks can capture more complex relationships in large datasets.

### Evalution Metrics:
* **Primay Metric:** Recall- We want to catch as many churners possible as even if it means including some false positives.
* **Secondary Metrics:** Precision - useful for minimizing the cost of wrongly targeting non-churners.
* **Why Recall is Primary:** Missing a real churner means lost revenue, while wrongly flagging a loyal customer results in a smaller cost (e.g., a discount)**

### Buisness Impact

1) High Recall, Moderate Precision = Catch more true churners, increase retention ROI.
2) Supports data-driven decision-making for retention strategy.
3) Identifies likely churners early.
