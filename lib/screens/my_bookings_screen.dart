import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../api.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';

class MyBookingsScreen extends StatefulWidget {
  const MyBookingsScreen({super.key});

  @override
  State<MyBookingsScreen> createState() => _MyBookingsScreenState();
}

class _MyBookingsScreenState extends State<MyBookingsScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _tabs;
  final _api = Api();
  var _loading = true;
  String? _error;

  List<Map<String, dynamic>> _bookings = [];

  List<Map<String, dynamic>> _filtered(String status) {
    if (status == 'All') return _bookings;
    return _bookings.where((b) => b['status'] == status).toList();
  }

  @override
  void initState() {
    super.initState();
    _tabs = TabController(length: 4, vsync: this);
    _loadBookings();
  }

  @override
  void dispose() {
    _tabs.dispose();
    super.dispose();
  }

  Future<void> _loadBookings() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final records = await _api.getBookings();
      if (!mounted) return;
      setState(() {
        _bookings = records
            .map(
              (record) =>
                  _bookingForDisplay(Map<String, dynamic>.from(record as Map)),
            )
            .toList();
        _loading = false;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _error = error.toString();
      });
    }
  }

  Map<String, dynamic> _bookingForDisplay(Map<String, dynamic> booking) {
    final status = booking['status'].toString().toLowerCase();
    final checkIn = DateTime.tryParse(booking['check_in'].toString());
    final checkOut = DateTime.tryParse(booking['check_out'].toString());
    final days = checkIn != null && checkOut != null
        ? checkOut.difference(checkIn).inDays
        : 0;
    final months = days <= 31 ? 1 : (days / 30).round();
    final statusLabel = '${status[0].toUpperCase()}${status.substring(1)}';

    return {
      'rawId': booking['id'],
      'id': 'SB-${booking['id']}',
      'hostel': booking['hostel_name']?.toString() ?? 'Hostel',
      'city': [booking['hostel_city'], booking['hostel_area']]
          .where((value) => value != null && value.toString().isNotEmpty)
          .join(' · '),
      'room': booking['room_id'] == null
          ? 'Hostel bed'
          : 'Room #${booking['room_id']}',
      'checkIn': checkIn == null
          ? booking['check_in'].toString()
          : '${checkIn.day.toString().padLeft(2, '0')} ${_month(checkIn.month)} ${checkIn.year}',
      'duration': '$months month${months == 1 ? '' : 's'}',
      'rent': (booking['single_room_price'] as num?)?.round() ?? 0,
      'status': statusLabel,
      'color': _statusColor(status),
      'bgColor': _statusBackground(status),
    };
  }

  String _month(int month) => const [
    'Jan',
    'Feb',
    'Mar',
    'Apr',
    'May',
    'Jun',
    'Jul',
    'Aug',
    'Sep',
    'Oct',
    'Nov',
    'Dec',
  ][month - 1];

  Color _statusColor(String status) {
    switch (status) {
      case 'confirmed':
        return AppTheme.green;
      case 'pending':
        return Colors.orange.shade700;
      case 'cancelled':
        return Colors.red.shade600;
      default:
        return const Color(0xFF185FA5);
    }
  }

  Color _statusBackground(String status) {
    switch (status) {
      case 'confirmed':
        return AppTheme.greenLight;
      case 'pending':
        return Colors.orange.shade50;
      case 'cancelled':
        return Colors.red.shade50;
      default:
        return const Color(0xFFE3EEF9);
    }
  }

  void _cancelBooking(Map<String, dynamic> booking) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: AppTheme.card,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Row(
          children: [
            Container(
              width: 38,
              height: 38,
              decoration: BoxDecoration(
                color: Colors.red.shade50,
                shape: BoxShape.circle,
              ),
              child: Icon(
                Icons.cancel_outlined,
                color: Colors.red.shade600,
                size: 20,
              ),
            ),
            const SizedBox(width: 10),
            Text(
              'Cancel Booking',
              style: GoogleFonts.playfairDisplay(
                fontSize: 16,
                fontWeight: FontWeight.w700,
                color: AppTheme.textPrimary,
              ),
            ),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              'Are you sure you want to cancel booking ${booking['id']} at ${booking['hostel']}?',
              style: GoogleFonts.dmSans(
                fontSize: 13,
                color: AppTheme.textSecondary,
                height: 1.5,
              ),
            ),
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.red.shade50,
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: Colors.red.shade200, width: 0.5),
              ),
              child: Row(
                children: [
                  Icon(
                    Icons.info_outline_rounded,
                    color: Colors.red.shade600,
                    size: 15,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'Cancellations may be subject to the hostel refund policy.',
                      style: GoogleFonts.dmSans(
                        fontSize: 11,
                        color: Colors.red.shade700,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text(
              'Keep Booking',
              style: GoogleFonts.dmSans(
                color: AppTheme.textMuted,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          ElevatedButton(
            onPressed: () async {
              Navigator.pop(context);
              try {
                await _api.cancelBooking(booking['rawId'] as int);
                await _loadBookings();
                if (!mounted) return;
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Text(
                      'Booking ${booking['id']} cancelled.',
                      style: GoogleFonts.dmSans(),
                    ),
                    backgroundColor: Colors.red.shade600,
                    behavior: SnackBarBehavior.floating,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    margin: const EdgeInsets.all(16),
                  ),
                );
              } catch (error) {
                if (!mounted) return;
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Text(
                      error.toString(),
                      style: GoogleFonts.dmSans(),
                    ),
                    backgroundColor: Colors.red.shade600,
                    behavior: SnackBarBehavior.floating,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    margin: const EdgeInsets.all(16),
                  ),
                );
              }
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.red.shade600,
              foregroundColor: Colors.white,
              elevation: 0,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(10),
              ),
            ),
            child: Text(
              'Yes, Cancel',
              style: GoogleFonts.dmSans(fontWeight: FontWeight.w600),
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: AppTheme.bg,
        appBar: AppBar(
          title: const Text('My Bookings'),
          leading: IconButton(
            icon: const Icon(Icons.arrow_back_rounded),
            onPressed: () => Navigator.pop(context),
          ),
          bottom: TabBar(
            controller: _tabs,
            labelColor: Colors.white,
            unselectedLabelColor: Colors.white60,
            indicatorColor: AppTheme.tealMint,
            tabs: const [
              Tab(text: 'All'),
              Tab(text: 'Pending'),
              Tab(text: 'Confirmed'),
              Tab(text: 'Past'),
            ],
          ),
        ),
        body: _loading
            ? const Center(
                child: CircularProgressIndicator(color: AppTheme.teal),
              )
            : _error != null
            ? _buildError()
            : TabBarView(
                controller: _tabs,
                children: [
                  _buildList(_filtered('All')),
                  _buildList(_filtered('Pending')),
                  _buildList(_filtered('Confirmed')),
                  _buildList([
                    ..._filtered('Completed'),
                    ..._filtered('Cancelled'),
                  ]),
                ],
              ),
      ),
    );
  }

  Widget _buildError() => Center(
    child: Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            Icons.error_outline_rounded,
            color: Colors.red.shade600,
            size: 44,
          ),
          const SizedBox(height: 12),
          Text(
            _error!,
            textAlign: TextAlign.center,
            style: GoogleFonts.dmSans(color: AppTheme.textSecondary),
          ),
          const SizedBox(height: 12),
          IconButton(
            tooltip: 'Retry',
            onPressed: _loadBookings,
            icon: const Icon(Icons.refresh_rounded),
            color: AppTheme.teal,
          ),
        ],
      ),
    ),
  );

  Widget _buildList(List<Map<String, dynamic>> items) {
    if (items.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.hotel_outlined,
              size: 60,
              color: AppTheme.textMuted.withOpacity(0.3),
            ),
            const SizedBox(height: 14),
            Text(
              'No bookings here',
              style: GoogleFonts.dmSans(
                color: AppTheme.textMuted,
                fontSize: 15,
              ),
            ),
            const SizedBox(height: 6),
            Text(
              'Your bookings will appear here',
              style: GoogleFonts.dmSans(
                color: AppTheme.textMuted,
                fontSize: 12,
              ),
            ),
          ],
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
      itemCount: items.length,
      itemBuilder: (_, i) => _bookingCard(items[i]),
    );
  }

  Widget _bookingCard(Map<String, dynamic> b) {
    final status = b['status'] as String;
    final color = b['color'] as Color;
    final bgColor = b['bgColor'] as Color;
    final isPending = status == 'Pending';
    final isConfirmed = status == 'Confirmed';
    final canCancel = isPending || isConfirmed;

    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppTheme.border, width: 0.5),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        children: [
          // Top
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 14, 16, 12),
            child: Row(
              children: [
                Container(
                  width: 46,
                  height: 46,
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                      colors: [AppTheme.tealDeep, AppTheme.teal],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(
                    Icons.apartment_rounded,
                    color: Colors.white,
                    size: 22,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        b['hostel'] as String,
                        style: GoogleFonts.dmSans(
                          fontSize: 15,
                          fontWeight: FontWeight.w700,
                          color: AppTheme.textPrimary,
                        ),
                      ),
                      const SizedBox(height: 3),
                      Row(
                        children: [
                          const Icon(
                            Icons.location_on_rounded,
                            size: 12,
                            color: AppTheme.textMuted,
                          ),
                          const SizedBox(width: 3),
                          Text(
                            b['city'] as String,
                            style: GoogleFonts.dmSans(
                              fontSize: 11,
                              color: AppTheme.textMuted,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 10,
                    vertical: 5,
                  ),
                  decoration: BoxDecoration(
                    color: bgColor,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    status,
                    style: GoogleFonts.dmSans(
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      color: color,
                    ),
                  ),
                ),
              ],
            ),
          ),

          Divider(height: 0.5, color: AppTheme.border, thickness: 0.5),

          // Details grid
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
            child: Row(
              children: [
                Expanded(
                  child: _detailCol(
                    Icons.meeting_room_rounded,
                    'Room',
                    b['room'] as String,
                  ),
                ),
                Expanded(
                  child: _detailCol(
                    Icons.event_rounded,
                    'Check-in',
                    b['checkIn'] as String,
                  ),
                ),
                Expanded(
                  child: _detailCol(
                    Icons.date_range_rounded,
                    'Duration',
                    b['duration'] as String,
                  ),
                ),
              ],
            ),
          ),

          // Booking ID + total
          Container(
            margin: const EdgeInsets.fromLTRB(12, 0, 12, 12),
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            decoration: BoxDecoration(
              color: AppTheme.bg,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: AppTheme.border, width: 0.5),
            ),
            child: Row(
              children: [
                const Icon(
                  Icons.receipt_long_rounded,
                  size: 14,
                  color: AppTheme.textMuted,
                ),
                const SizedBox(width: 6),
                Text(
                  b['id'] as String,
                  style: GoogleFonts.dmSans(
                    fontSize: 12,
                    color: AppTheme.textMuted,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const Spacer(),
                Text(
                  'PKR ${(b['rent'] as int).toString().replaceAllMapped(RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))'), (m) => "${m[1]},")} / month',
                  style: GoogleFonts.dmSans(
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.teal,
                  ),
                ),
              ],
            ),
          ),

          // Action buttons
          if (canCancel)
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
              child: Row(
                children: [
                  // View Details
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () {},
                      icon: const Icon(Icons.info_outline_rounded, size: 15),
                      label: Text(
                        'Details',
                        style: GoogleFonts.dmSans(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      style: OutlinedButton.styleFrom(
                        foregroundColor: AppTheme.teal,
                        side: const BorderSide(
                          color: AppTheme.teal,
                          width: 0.5,
                        ),
                        padding: const EdgeInsets.symmetric(vertical: 10),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 10),
                  // Cancel
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: () => _cancelBooking(b),
                      icon: const Icon(Icons.cancel_outlined, size: 15),
                      label: Text(
                        'Cancel Booking',
                        style: GoogleFonts.dmSans(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.red.shade600,
                        foregroundColor: Colors.white,
                        elevation: 0,
                        padding: const EdgeInsets.symmetric(vertical: 10),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  Widget _detailCol(IconData icon, String label, String value) {
    return Column(
      children: [
        Icon(icon, size: 16, color: AppTheme.teal),
        const SizedBox(height: 4),
        Text(
          value,
          style: GoogleFonts.dmSans(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: AppTheme.textPrimary,
          ),
          textAlign: TextAlign.center,
        ),
        Text(
          label,
          style: GoogleFonts.dmSans(fontSize: 10, color: AppTheme.textMuted),
        ),
      ],
    );
  }
}
