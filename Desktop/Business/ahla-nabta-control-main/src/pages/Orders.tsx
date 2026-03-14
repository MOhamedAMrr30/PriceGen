import { useState, useMemo, useEffect } from "react";
import { useClients, useProducts, useClientPricing, useCreateOrder, useDeleteOrder, useUpdateOrder } from "@/hooks/use-data";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { MarginBadge, getMarginZone } from "@/components/MarginBadge";
import { generateInvoicePDF, generateDeliveryNotePDF, loadLogoBase64 } from "@/lib/pdf";
import { Plus, Trash2, FileText, MessageCircle, ShoppingCart, Edit3, CalendarDays, Pencil } from "lucide-react";
import { toast } from "@/hooks/use-toast";
import { useOrders, useOrderWithItems } from "@/hooks/use-data";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

interface OrderLineItem {
  product_id: string;
  productName: string;
  unit: string;
  quantity: number;
  selling_price: number;
  cost: number;
  package_size?: number;
  num_packages?: number;
}

const CreateOrderForm = ({ onClose, editingOrderId }: { onClose: () => void; editingOrderId?: string }) => {
  const { data: clients } = useClients();
  const { data: products } = useProducts();
  const { data: existingOrders } = useOrders();
  const createOrder = useCreateOrder();
  const updateOrder = useUpdateOrder();

  // Restore last order from localStorage
  const lastOrder = useMemo(() => {
    try {
      const saved = localStorage.getItem("ahla_last_order");
      return saved ? JSON.parse(saved) : null;
    } catch { return null; }
  }, []);

  const [clientId, setClientId] = useState(lastOrder?.clientId || "");
  const [orderDate, setOrderDate] = useState(new Date().toISOString().split("T")[0]);
  const [deliveryDate, setDeliveryDate] = useState(new Date().toISOString().split("T")[0]);
  const [transportCost, setTransportCost] = useState(lastOrder?.transportCost || "0");
  const [packagingCost, setPackagingCost] = useState(lastOrder?.packagingCost || "0");
  const [items, setItems] = useState<OrderLineItem[]>(lastOrder?.items || []);
  const [selectedProduct, setSelectedProduct] = useState("");
  const [qty, setQty] = useState("1");
  const [packageSize, setPackageSize] = useState("0.1");
  const [numPackages, setNumPackages] = useState("1");
  const [clientPrices, setClientPrices] = useState<Record<string, number>>({});
  const [loadingOrder, setLoadingOrder] = useState(false);
  const [clientOrderCount, setClientOrderCount] = useState<number>(0);

  // Detect if selected client is a supermarket
  const selectedClient = clients?.find((c) => c.id === clientId);
  const isSupermarket = selectedClient?.type === "supermarket";

  // Load items from a previous order into the form
  const loadFromOrder = async (orderId: string) => {
    if (!orderId) return;
    setLoadingOrder(true);
    try {
      const [orderRes, itemsRes] = await Promise.all([
        supabase.from("orders").select("*").eq("id", orderId).maybeSingle(),
        supabase.from("order_items").select("*, products(name, unit, cost_per_unit)").eq("order_id", orderId),
      ]);
      if (orderRes.error) throw orderRes.error;
      if (itemsRes.error) throw itemsRes.error;
      const order = orderRes.data;
      const orderItems = itemsRes.data;
      if (order) {
        setClientId(order.client_id);
        setOrderDate(order.order_date || new Date().toISOString().split("T")[0]);
        setDeliveryDate(order.delivery_date || new Date().toISOString().split("T")[0]);
        setTransportCost(String(order.transport_cost || 0));
        setPackagingCost(String(order.packaging_cost || 0));
        // Load client prices for the selected client
        const { data: priceData } = await supabase.from("client_pricing").select("product_id, selling_price").eq("client_id", order.client_id);
        const map: Record<string, number> = {};
        priceData?.forEach((p) => { map[p.product_id] = p.selling_price; });
        setClientPrices(map);
      }
      if (orderItems) {
        setItems(orderItems.map((i: any) => {
          // Recover package info for kg items
          let pkgSize: number | undefined;
          let numPkgs: number | undefined;
          if (i.products?.unit === "kg" && i.selling_price_used > 0 && i.total_revenue > 0) {
            numPkgs = Math.round(i.total_revenue / i.selling_price_used);
            pkgSize = numPkgs > 0 ? i.quantity / numPkgs : undefined;
          }
          const name = i.products?.name || "Unknown";
          return {
            product_id: i.product_id,
            productName: pkgSize && numPkgs ? `${name} (${pkgSize}kg × ${numPkgs})` : name,
            unit: i.products?.unit || "pc",
            quantity: i.quantity,
            selling_price: i.selling_price_used,
            cost: i.cost_used ?? i.products?.cost_per_unit ?? 0,
            package_size: pkgSize,
            num_packages: numPkgs,
          };
        }));
      }
      toast({ title: "Order loaded", description: "Edit the details and save as a new order" });
    } catch (err) {
      toast({ title: "Failed to load order", variant: "destructive" });
    } finally {
      setLoadingOrder(false);
    }
  };

  // Auto-load client prices if we restored a clientId
  useEffect(() => {
    if (editingOrderId) {
      loadFromOrder(editingOrderId);
    } else if (lastOrder?.clientId) {
      loadClientPrices(lastOrder.clientId);
    }
  }, []);

  const loadClientPrices = async (cid: string) => {
    setClientId(cid);
    const { data } = await supabase.from("client_pricing").select("product_id, selling_price").eq("client_id", cid);
    const map: Record<string, number> = {};
    data?.forEach((p) => { map[p.product_id] = p.selling_price; });
    setClientPrices(map);
    // Update existing items with the new client's prices
    setItems(prev => prev.map(item => {
      const newPrice = map[item.product_id];
      if (newPrice !== undefined) {
        return { ...item, selling_price: newPrice };
      }
      return item;
    }));
    // Fetch total order count for this client
    const { count } = await supabase.from("orders").select("id", { count: "exact", head: true }).eq("client_id", cid);
    setClientOrderCount(count || 0);
  };

  const selectedProd = products?.find((p) => p.id === selectedProduct);
  const isKg = selectedProd?.unit === "kg";

  const addItem = () => {
    if (!selectedProduct) return;
    const prod = selectedProd;
    if (!prod) return;
    const price = clientPrices[selectedProduct];
    if (!price) {
      toast({ title: "No price set", description: `Set a price for ${prod.name} for this client first`, variant: "destructive" });
      return;
    }
    // For kg products: total qty = packageSize * numPackages
    // For pc products: total qty = qty
    const pkgSz = parseFloat(packageSize) || 0;
    const numPkgs = parseInt(numPackages) || 0;
    const totalQty = isKg
      ? pkgSz * numPkgs
      : parseFloat(qty) || 0;
    if (totalQty <= 0) {
      toast({ title: "Invalid quantity", variant: "destructive" });
      return;
    }
    setItems([...items, {
      product_id: selectedProduct,
      productName: isKg ? `${prod.name} (${packageSize}kg × ${numPackages})` : prod.name,
      unit: prod.unit,
      quantity: totalQty,
      selling_price: price,
      cost: prod.cost_per_unit,
      package_size: isKg ? pkgSz : undefined,
      num_packages: isKg ? numPkgs : undefined,
    }]);
    setSelectedProduct("");
    setQty("1");
    setPackageSize("0.1");
    setNumPackages("1");
  };

  const removeItem = (idx: number) => setItems(items.filter((_, i) => i !== idx));

  const updateItem = (idx: number, field: keyof OrderLineItem, value: string) => {
    setItems(items.map((item, i) => {
      if (i !== idx) return item;
      const numVal = parseFloat(value) || 0;
      if (field === "quantity" || field === "selling_price" || field === "cost") {
        return { ...item, [field]: numVal };
      }
      return { ...item, [field]: value };
    }));
  };

  // For supermarket: revenue = price × num_packages (price is per-pack)
  // For restaurant: revenue = price × quantity (price is per-kg)
  const itemRevenue = (item: OrderLineItem) => {
    if (isSupermarket && item.num_packages) {
      return item.selling_price * item.num_packages;
    }
    return item.selling_price * item.quantity;
  };

  const calc = useMemo(() => {
    const revenue = items.reduce((s, i) => s + itemRevenue(i), 0);
    const farmerCost = items.reduce((s, i) => s + i.cost * i.quantity, 0);
    const tc = parseFloat(transportCost) || 0;
    const pc = parseFloat(packagingCost) || 0;
    const logistics = tc + pc;
    const totalCost = farmerCost + logistics;
    const profit = revenue - totalCost;
    const margin = revenue > 0 ? (profit / revenue) * 100 : 0;
    return { revenue, farmerCost, logistics, totalCost, profit, margin };
  }, [items, transportCost, packagingCost, isSupermarket]);

  const handleSave = () => {
    if (!clientId) { toast({ title: "Select a client", variant: "destructive" }); return; }
    if (!items.length) { toast({ title: "Add at least one item", variant: "destructive" }); return; }
    if (deliveryDate < orderDate) { toast({ title: "Delivery date must be ≥ order date", variant: "destructive" }); return; }
    const tc = parseFloat(transportCost) || 0;
    if (tc < 0) { toast({ title: "Transport cost cannot be negative", variant: "destructive" }); return; }

    // Save to localStorage for next time
    localStorage.setItem("ahla_last_order", JSON.stringify({
      clientId,
      transportCost,
      packagingCost,
      items,
    }));

    const orderData = {
      client_id: clientId,
      order_date: orderDate,
      delivery_date: deliveryDate,
      transport_cost: tc,
      packaging_cost: parseFloat(packagingCost) || 0,
      total_revenue: calc.revenue,
      total_cost: calc.totalCost,
      net_profit: calc.profit,
      margin_percentage: Math.round(calc.margin * 100) / 100,
      margin_zone: getMarginZone(calc.margin),
    };

    const orderItems = items.map((i) => ({
      order_id: editingOrderId || "",
      product_id: i.product_id,
      quantity: i.quantity,
      selling_price_used: i.selling_price,
      cost_used: i.cost,
      total_revenue: itemRevenue(i),
      total_cost: i.cost * i.quantity,
    }));

    if (editingOrderId) {
      // Update existing order
      updateOrder.mutate(
        { orderId: editingOrderId, order: orderData, items: orderItems },
        {
          onSuccess: () => {
            toast({ title: "Order updated" });
            onClose();
          },
        }
      );
    } else {
      // Create new order
      createOrder.mutate(
        { order: orderData, items: orderItems },
        {
          onSuccess: () => {
            toast({ title: "Order created" });
            onClose();
          },
        }
      );
    }
  };

  const zone = getMarginZone(calc.margin);

  return (
    <div className="space-y-6">
      {/* Load from previous order */}
      {existingOrders && existingOrders.length > 0 && (
        <div className="flex items-end gap-2 p-3 rounded-lg border border-dashed border-primary/30 bg-primary/5">
          <div className="flex-1">
            <Label className="text-xs text-muted-foreground">Load from previous order</Label>
            <Select onValueChange={loadFromOrder} disabled={loadingOrder}>
              <SelectTrigger>
                <SelectValue placeholder={loadingOrder ? "Loading..." : "Select an order to edit from..."} />
              </SelectTrigger>
              <SelectContent>
                {existingOrders.map((o: any) => (
                  <SelectItem key={o.id} value={o.id}>
                    {o.clients?.name} — {o.delivery_date} — {Number(o.total_revenue).toFixed(2)} EGP
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <Edit3 className="h-4 w-4 text-primary mb-2.5" />
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <Label>Client</Label>
          <Select value={clientId} onValueChange={loadClientPrices}>
            <SelectTrigger><SelectValue placeholder="Select client" /></SelectTrigger>
            <SelectContent>
              {clients?.map((c) => <SelectItem key={c.id} value={c.id}>{c.name} {c.type === "supermarket" ? "🏪" : "🍽️"}</SelectItem>)}
            </SelectContent>
          </Select>
          {clientId && (
            <div className="flex items-center gap-2 mt-1.5">
              {isSupermarket && (
                <span className="text-xs px-2 py-0.5 rounded-full bg-blue-100 text-blue-700 font-medium">Supermarket</span>
              )}
              <span className="text-xs text-muted-foreground">Total orders: <strong>{clientOrderCount}</strong></span>
            </div>
          )}
        </div>
        <div></div>
        <div><Label>Order Date</Label><Input type="date" value={orderDate} onChange={(e) => setOrderDate(e.target.value)} /></div>
        <div><Label>Delivery Date</Label><Input type="date" value={deliveryDate} onChange={(e) => setDeliveryDate(e.target.value)} /></div>
        <div><Label>Transport Cost</Label><Input type="number" min="0" value={transportCost} onChange={(e) => setTransportCost(e.target.value)} /></div>
        <div><Label>Packaging Cost</Label><Input type="number" min="0" value={packagingCost} onChange={(e) => setPackagingCost(e.target.value)} /></div>
      </div>

      <div className="flex gap-2 items-end">
        <div className="flex-1">
          <Label>Product</Label>
          <Select value={selectedProduct} onValueChange={setSelectedProduct}>
            <SelectTrigger><SelectValue placeholder="Select product" /></SelectTrigger>
            <SelectContent>
              {products?.map((p) => <SelectItem key={p.id} value={p.id}>{p.name} ({p.unit})</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
        {isKg ? (
          <>
            <div className="w-24">
              <Label>Pkg Size (kg)</Label>
              <Input type="number" min="0.01" step="0.01" value={packageSize} onChange={(e) => setPackageSize(e.target.value)} />
            </div>
            <div className="w-20">
              <Label>× Pkgs</Label>
              <Input type="number" min="1" step="1" value={numPackages} onChange={(e) => setNumPackages(e.target.value)} />
            </div>
            <div className="text-xs text-muted-foreground self-center pt-4 w-16">
              = {((parseFloat(packageSize) || 0) * (parseInt(numPackages) || 0)).toFixed(2)} kg
            </div>
          </>
        ) : (
          <div className="w-24">
            <Label>Qty</Label>
            <Input type="number" min="1" value={qty} onChange={(e) => setQty(e.target.value)} />
          </div>
        )}
        <Button onClick={addItem} size="icon"><Plus className="h-4 w-4" /></Button>
      </div>

      {items.length > 0 && (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Product</TableHead>
              {isSupermarket && <TableHead className="w-20">Pkg Size</TableHead>}
              <TableHead className="w-24">{isSupermarket ? "Pckgs" : "Qty"}</TableHead>
              <TableHead className="w-24">{isSupermarket ? "Price/Pack" : "Price"}</TableHead>
              <TableHead className="w-24">Base Cost</TableHead>
              <TableHead>Revenue</TableHead>
              <TableHead></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {items.map((item, idx) => {
              const totalWeight = isSupermarket && item.unit === "kg" && item.package_size && item.num_packages
                ? item.package_size * item.num_packages
                : item.quantity;
              const baseCost = item.cost * totalWeight;
              return (
                <TableRow key={idx}>
                  <TableCell>{item.productName}</TableCell>
                  {isSupermarket && (
                    <TableCell>
                      {item.unit === "kg" ? (
                        <Input
                          type="number"
                          min="0.01"
                          step="0.01"
                          className="h-8 w-16"
                          value={item.package_size || ""}
                          placeholder="kg"
                          onChange={(e) => {
                            const newPkgSz = parseFloat(e.target.value) || 0;
                            setItems(items.map((it, i) => i !== idx ? it : {
                              ...it,
                              package_size: newPkgSz,
                              quantity: newPkgSz * (it.num_packages || 1),
                              productName: it.productName.includes("(")
                                ? it.productName.replace(/\(.*\)/, `(${newPkgSz}kg × ${it.num_packages || 1})`)
                                : `${it.productName} (${newPkgSz}kg × ${it.num_packages || 1})`,
                            }));
                          }}
                        />
                      ) : (
                        <span className="text-xs text-muted-foreground">1 pc</span>
                      )}
                    </TableCell>
                  )}
                  <TableCell>
                    <Input
                      type="number"
                      min="0"
                      step="1"
                      className="h-8 w-20"
                      value={isSupermarket ? (item.num_packages || item.quantity) : item.quantity}
                      onChange={(e) => {
                        if (isSupermarket) {
                          const newPkgs = parseInt(e.target.value) || 0;
                          const pkgSz = item.package_size || 0;
                          setItems(items.map((it, i) => i !== idx ? it : {
                            ...it,
                            num_packages: newPkgs,
                            quantity: it.unit === "kg" ? (pkgSz * newPkgs) : newPkgs,
                            productName: it.package_size
                              ? (it.productName.includes("(")
                                ? it.productName.replace(/\(.*\)/, `(${it.package_size}kg × ${newPkgs})`)
                                : `${it.productName} (${it.package_size}kg × ${newPkgs})`)
                              : it.productName,
                          }));
                        } else {
                          updateItem(idx, "quantity", e.target.value);
                        }
                      }}
                    />
                  </TableCell>
                  <TableCell>
                    <Input
                      type="number"
                      min="0"
                      step="0.01"
                      className="h-8 w-20"
                      value={item.selling_price}
                      onChange={(e) => updateItem(idx, "selling_price", e.target.value)}
                    />
                  </TableCell>
                  <TableCell>
                    <div>
                      <Input
                        type="number"
                        min="0"
                        step="0.01"
                        className="h-8 w-20"
                        value={item.cost}
                        onChange={(e) => updateItem(idx, "cost", e.target.value)}
                      />
                      <span className="block text-xs text-muted-foreground mt-0.5">
                        = {baseCost.toFixed(2)} ({item.cost}/{item.unit} × {totalWeight})
                      </span>
                    </div>
                  </TableCell>
                  <TableCell>{itemRevenue(item).toFixed(2)}</TableCell>
                  <TableCell><Button variant="ghost" size="icon" onClick={() => removeItem(idx)}><Trash2 className="h-4 w-4 text-destructive" /></Button></TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      )}

      <Card className={cn(
        "border-2",
        zone === "green" && "border-margin-green/30 bg-margin-green/5",
        zone === "yellow" && "border-margin-yellow/30 bg-margin-yellow/5",
        zone === "red" && "border-margin-red/30 bg-margin-red/5",
      )}>
        <CardContent className="pt-4 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-4 text-sm">
          <div><span className="text-muted-foreground">Revenue</span><p className="font-bold text-lg">{calc.revenue.toFixed(2)}</p></div>
          <div><span className="text-muted-foreground">Farmer Cost</span><p className="font-bold text-lg text-orange-600">{calc.farmerCost.toFixed(2)}</p></div>
          <div><span className="text-muted-foreground">Transport + Pkg</span><p className="font-bold text-lg">{calc.logistics.toFixed(2)}</p></div>
          <div><span className="text-muted-foreground">Total Cost</span><p className="font-bold text-lg">{calc.totalCost.toFixed(2)}</p></div>
          <div><span className="text-muted-foreground">Net Profit</span><p className="font-bold text-lg">{calc.profit.toFixed(2)}</p></div>
          <div><span className="text-muted-foreground">Margin</span><p className="text-lg"><MarginBadge margin={calc.margin} /></p></div>
        </CardContent>
      </Card>

      <Button onClick={handleSave} className="w-full" disabled={createOrder.isPending || updateOrder.isPending}>
        <ShoppingCart className="mr-2 h-4 w-4" />{editingOrderId ? "Update Order" : "Save Order"}
      </Button>
    </div>
  );
};

const OrderRow = ({ order, index, onEdit }: { order: any; index: number; onEdit: (orderId: string) => void }) => {
  const [showDetail, setShowDetail] = useState(false);
  const { data: detail } = useOrderWithItems(showDetail ? order.id : "");
  const removeOrder = useDeleteOrder();
  const [logoBase64, setLogoBase64] = useState<string | undefined>();

  useEffect(() => {
    loadLogoBase64().then(setLogoBase64);
  }, []);

  const orderNum = index;

  const clientType = order.clients?.type || "restaurant";
  const isSuper = clientType === "supermarket";

  const handleInvoice = () => {
    if (!detail) return;
    const pdf = generateInvoicePDF({
      clientName: detail.order?.clients?.name || "",
      orderDate: detail.order?.order_date || "",
      deliveryDate: detail.order?.delivery_date || "",
      isSupermarket: isSuper,
      items: detail.items?.map((i: any) => {
        // Try to parse pack info from product name pattern "Name (0.1kg × 5)"
        const match = (i.products?.name || "").match(/^(.+)$/) ? null : null;
        const nameStr = i.products?.name || "";
        // Check the quantity and price pattern to detect pack info stored in product name of the order form
        // We reconstruct from the total_revenue / selling_price if supermarket
        let packageSize: number | undefined;
        let numPackages: number | undefined;
        if (isSuper && i.selling_price_used > 0) {
          numPackages = Math.round(i.total_revenue / i.selling_price_used);
          packageSize = numPackages > 0 ? i.quantity / numPackages : undefined;
        }
        return {
          product: nameStr,
          qty: i.quantity,
          unit: i.products?.unit || "",
          price: i.selling_price_used,
          total: i.total_revenue,
          packageSize,
          numPackages,
        };
      }) || [],
      totalRevenue: detail.order?.total_revenue || 0,
      logoBase64,
    });
    pdf.save(`Ahla Nabta Invoice ${orderNum}.pdf`);
  };

  const handleDeliveryNote = () => {
    if (!detail) return;
    const pdf = generateDeliveryNotePDF({
      clientName: detail.order?.clients?.name || "",
      orderDate: detail.order?.order_date || "",
      deliveryDate: detail.order?.delivery_date || "",
      isSupermarket: isSuper,
      items: detail.items?.map((i: any) => {
        let packageSize: number | undefined;
        let numPackages: number | undefined;
        if (isSuper && i.selling_price_used > 0) {
          numPackages = Math.round(i.total_revenue / i.selling_price_used);
          packageSize = numPackages > 0 ? i.quantity / numPackages : undefined;
        }
        return {
          product: i.products?.name || "",
          qty: i.quantity,
          unit: i.products?.unit || "",
          price: i.selling_price_used,
          total: i.total_revenue,
          baseCost: Number(i.cost_used || 0),
          totalBaseCost: Number(i.total_cost || 0),
          packageSize,
          numPackages,
        };
      }) || [],
      totalRevenue: detail.order?.total_revenue || 0,
      logoBase64,
    });
    pdf.save(`Ahla Nabta Delivery Note ${orderNum}.pdf`);
  };

  const handleWhatsApp = () => {
    const clientName = order.clients?.name || "Client";
    const msg = encodeURIComponent(`Dear ${clientName}, please find attached your invoice for ${order.delivery_date}.`);
    const phone = order.clients?.phone ? order.clients.phone.replace(/\D/g, "") : "";
    window.open(`https://wa.me/${phone}?text=${msg}`, "_blank");
  };

  return (
    <>
      <TableRow className="cursor-pointer" onClick={() => setShowDetail(!showDetail)}>
        <TableCell className="font-medium">
          {order.clients?.name}
          <span className="ml-2 text-xs font-semibold text-primary bg-primary/10 px-1.5 py-0.5 rounded-full">#{orderNum}</span>
        </TableCell>
        <TableCell>{order.order_date}</TableCell>
        <TableCell>{order.delivery_date}</TableCell>
        <TableCell>{Number(order.total_revenue).toFixed(2)}</TableCell>
        <TableCell>{Number(order.net_profit).toFixed(2)}</TableCell>
        <TableCell><MarginBadge margin={Number(order.margin_percentage)} /></TableCell>
        <TableCell>
          <div className="flex gap-1">
            <Button variant="ghost" size="icon" className="h-7 w-7" onClick={(e) => { e.stopPropagation(); onEdit(order.id); }}>
              <Pencil className="h-3 w-3 text-primary" />
            </Button>
            <Button variant="ghost" size="icon" className="h-7 w-7" onClick={(e) => { e.stopPropagation(); removeOrder.mutate(order.id); }}>
              <Trash2 className="h-3 w-3 text-destructive" />
            </Button>
          </div>
        </TableCell>
      </TableRow>
      {showDetail && detail && (
        <TableRow>
          <TableCell colSpan={7} className="bg-muted/30 p-4">
            <div className="space-y-3">
              <Table>
                <TableHeader><TableRow>
                  <TableHead>Product</TableHead>
                  <TableHead>{isSuper ? "Pckgs" : "Qty"}</TableHead>
                  <TableHead>{isSuper ? "Price/Pack" : "Sell Price"}</TableHead>
                  <TableHead>Base Cost</TableHead>
                  <TableHead>Revenue</TableHead>
                  <TableHead>Farmer Total</TableHead>
                </TableRow></TableHeader>
                <TableBody>
                  {detail.items?.map((i: any) => {
                    let numPkgs = 0;
                    let pkgSz = 0;
                    if (isSuper && i.selling_price_used > 0) {
                      numPkgs = Math.round(i.total_revenue / i.selling_price_used);
                      pkgSz = numPkgs > 0 ? (i.quantity / numPkgs) : 0;
                    }
                    return (
                      <TableRow key={i.id}>
                        <TableCell>
                          {i.products?.name}
                          {isSuper && numPkgs > 0 && pkgSz > 0 && (
                            <span className="block text-xs text-muted-foreground">{pkgSz}kg × {numPkgs} pckgs</span>
                          )}
                        </TableCell>
                        <TableCell>
                          {isSuper
                            ? (numPkgs > 0 ? numPkgs : i.quantity)
                            : `${i.quantity} ${i.products?.unit}`}
                        </TableCell>
                        <TableCell>{Number(i.selling_price_used).toFixed(2)}{isSuper ? "/pack" : ""}</TableCell>
                        <TableCell className="text-orange-600">
                          {Number(i.cost_used).toFixed(2)}/{i.products?.unit}
                        </TableCell>
                        <TableCell>{Number(i.total_revenue).toFixed(2)}</TableCell>
                        <TableCell className="font-bold text-orange-600">
                          {Number(i.total_cost).toFixed(2)}
                          <span className="block text-xs font-normal text-muted-foreground">
                            {Number(i.cost_used).toFixed(2)} × {i.quantity}{i.products?.unit}
                          </span>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>

              {/* Cost Breakdown */}
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 p-3 rounded-md border bg-background">
                <div>
                  <span className="text-xs text-muted-foreground">Base Cost (Farmer)</span>
                  <p className="font-bold text-orange-600">
                    {detail.items?.reduce((s: number, i: any) => s + Number(i.total_cost), 0).toFixed(2)}
                  </p>
                </div>
                <div>
                  <span className="text-xs text-muted-foreground">Transport</span>
                  <p className="font-bold text-blue-600">
                    {Number(detail.order?.transport_cost || 0).toFixed(2)}
                  </p>
                </div>
                <div>
                  <span className="text-xs text-muted-foreground">Packaging</span>
                  <p className="font-bold text-purple-600">
                    {Number(detail.order?.packaging_cost || 0).toFixed(2)}
                  </p>
                </div>
                <div>
                  <span className="text-xs text-muted-foreground">Total Cost</span>
                  <p className="font-bold text-red-600">
                    {((detail.items?.reduce((s: number, i: any) => s + Number(i.total_cost), 0) || 0) + Number(detail.order?.transport_cost || 0) + Number(detail.order?.packaging_cost || 0)).toFixed(2)}
                  </p>
                </div>
                <div>
                  <span className="text-xs text-muted-foreground">Revenue</span>
                  <p className="font-bold text-green-600">
                    {Number(detail.order?.total_revenue || 0).toFixed(2)}
                  </p>
                </div>
                <div>
                  <span className="text-xs text-muted-foreground">Net Profit</span>
                  <p className={cn("font-bold", Number(detail.order?.net_profit || 0) >= 0 ? "text-green-600" : "text-red-600")}>
                    {Number(detail.order?.net_profit || 0).toFixed(2)}
                  </p>
                </div>
              </div>

              <div className="flex flex-wrap gap-2">
                <Button size="sm" variant="outline" onClick={() => onEdit(order.id)}><Pencil className="mr-1 h-3 w-3" />Edit Order</Button>
                <Button size="sm" variant="outline" onClick={handleInvoice}><FileText className="mr-1 h-3 w-3" />Invoice PDF</Button>
                <Button size="sm" variant="outline" onClick={handleDeliveryNote}><FileText className="mr-1 h-3 w-3" />Delivery Note</Button>
                <Button size="sm" variant="outline" onClick={handleWhatsApp}><MessageCircle className="mr-1 h-3 w-3" />WhatsApp</Button>
              </div>
            </div>
          </TableCell>
        </TableRow>
      )}
    </>
  );
};

const Orders = () => {
  const { data: orders, isLoading } = useOrders();
  const [creating, setCreating] = useState(false);
  const [editingOrderId, setEditingOrderId] = useState<string | undefined>();

  const handleOpenEdit = (orderId: string) => {
    setEditingOrderId(orderId);
    setCreating(true);
  };

  const handleCloseDialog = () => {
    setCreating(false);
    setEditingOrderId(undefined);
  };

  // Group orders by delivery_date
  const groupedOrders = useMemo(() => {
    if (!orders?.length) return [];
    const groups: { date: string; orders: any[]; totalRevenue: number; totalProfit: number; totalTransport: number; totalPackaging: number }[] = [];
    const map = new Map<string, any[]>();
    orders.forEach((o: any) => {
      const d = o.delivery_date || "Unknown";
      if (!map.has(d)) map.set(d, []);
      map.get(d)!.push(o);
    });
    map.forEach((orderList, date) => {
      groups.push({
        date,
        orders: orderList,
        totalRevenue: orderList.reduce((s, o) => s + Number(o.total_revenue || 0), 0),
        totalProfit: orderList.reduce((s, o) => s + Number(o.net_profit || 0), 0),
        totalTransport: orderList.reduce((s, o) => s + Number(o.transport_cost || 0), 0),
        totalPackaging: orderList.reduce((s, o) => s + Number(o.packaging_cost || 0), 0),
      });
    });
    return groups;
  }, [orders]);

  // Compute per-client order number (oldest order = #1)
  const clientOrderNumMap = useMemo(() => {
    if (!orders?.length) return new Map<string, number>();
    // Sort all orders by order_date ascending (oldest first), then by id for stability
    const sorted = [...orders].sort((a: any, b: any) => {
      const dateA = a.order_date || "";
      const dateB = b.order_date || "";
      if (dateA !== dateB) return dateA.localeCompare(dateB);
      return (a.id || "").localeCompare(b.id || "");
    });
    const counters = new Map<string, number>();
    const result = new Map<string, number>();
    sorted.forEach((o: any) => {
      const clientId = o.client_id || "";
      const count = (counters.get(clientId) || 0) + 1;
      counters.set(clientId, count);
      result.set(o.id, count);
    });
    return result;
  }, [orders]);

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between mb-4 sm:mb-6 gap-2">
        <h1 className="text-xl sm:text-2xl font-bold">Orders</h1>
        <Dialog open={creating} onOpenChange={(open) => { if (!open) handleCloseDialog(); else setCreating(true); }}>
          <DialogTrigger asChild>
            <Button><Plus className="mr-2 h-4 w-4" />Create Order</Button>
          </DialogTrigger>
          <DialogContent className="max-w-[95vw] sm:max-w-3xl max-h-[90vh] overflow-y-auto">
            <DialogHeader><DialogTitle>{editingOrderId ? "Edit Order" : "Create Order"}</DialogTitle></DialogHeader>
            <CreateOrderForm key={editingOrderId || "new"} onClose={handleCloseDialog} editingOrderId={editingOrderId} />
          </DialogContent>
        </Dialog>
      </div>
      <Card>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Client</TableHead>
                  <TableHead>Order Date</TableHead>
                  <TableHead>Delivery</TableHead>
                  <TableHead>Revenue</TableHead>
                  <TableHead>Profit</TableHead>
                  <TableHead>Margin</TableHead>
                  <TableHead></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading ? (
                  <TableRow><TableCell colSpan={7} className="text-center text-muted-foreground">Loading...</TableCell></TableRow>
                ) : !orders?.length ? (
                  <TableRow><TableCell colSpan={7} className="text-center text-muted-foreground">No orders yet</TableCell></TableRow>
                ) : (
                  groupedOrders.map((group) => {
                    const rows = group.orders.map((o: any) => {
                      const clientNum = clientOrderNumMap.get(o.id) || 0;
                      return <OrderRow key={o.id} order={o} index={clientNum} onEdit={handleOpenEdit} />;
                    });
                    return [
                      <TableRow key={`date-${group.date}`} className="bg-muted/50 hover:bg-muted/50">
                        <TableCell colSpan={3} className="font-semibold text-sm">
                          <div className="flex items-center gap-2">
                            <CalendarDays className="h-4 w-4 text-primary" />
                            {group.date}
                            <span className="text-xs text-muted-foreground font-normal">({group.orders.length} order{group.orders.length > 1 ? "s" : ""})</span>
                          </div>
                        </TableCell>
                        <TableCell className="font-semibold text-sm">{group.totalRevenue.toFixed(2)}</TableCell>
                        <TableCell className="font-semibold text-sm">{group.totalProfit.toFixed(2)}</TableCell>
                        <TableCell className="text-xs text-muted-foreground">
                          Transport: {group.totalTransport.toFixed(2)}
                        </TableCell>
                        <TableCell></TableCell>
                      </TableRow>,
                      ...rows,
                    ];
                  })
                )}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Orders;
