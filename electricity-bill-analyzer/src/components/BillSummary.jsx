function BillSummary({ userInfo, currentBill }) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-semibold mb-4">Account Information</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div>
              <h4 className="text-sm font-medium text-gray-500">Account Holder</h4>
              <p className="text-lg font-medium">{userInfo.customer_name}</p>
            </div>
            
            <div>
              <h4 className="text-sm font-medium text-gray-500">Account Number</h4>
              <p className="text-lg font-medium">{userInfo.account_number}</p>
            </div>
          </div>
          
          <div className="space-y-4">
            <div>
              <h4 className="text-sm font-medium text-gray-500">Current Usage</h4>
              <p className="text-lg font-medium">{currentBill.kwh_used} kWh</p>
            </div>
            
            <div>
              <h4 className="text-sm font-medium text-gray-500">Current Bill Amount</h4>
              <p className="text-lg font-medium">${currentBill.total_bill_amount}</p>
            </div>
          </div>
        </div>
      </div>
    );
  }
  
  export default BillSummary;