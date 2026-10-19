import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:video_player/video_player.dart';

class ParentFeesScreen extends StatefulWidget {
  const ParentFeesScreen({super.key});

  @override
  State<ParentFeesScreen> createState() => _ParentFeesScreenState();
}

class _ParentFeesScreenState extends State<ParentFeesScreen> {
  late VideoPlayerController _videoController;

  static const Color background = Color(0xFF063E40);
  static const Color darkTeal = Color(0xFF074F51);
  static const Color teal = Color(0xFF0A6B6E);
  static const Color mint = Color(0xFF5DCAA5);
  static const Color green = Color(0xFF35EE96);
  static const Color warning = Color(0xFFFFD166);

  @override
  void initState() {
    super.initState();

    _videoController = VideoPlayerController.asset(
      'assets/images/background.mp4',
    )
      ..initialize().then((_) {
        if (!mounted) return;

        _videoController
          ..setLooping(true)
          ..setVolume(0)
          ..play();

        setState(() {});
      });
  }

  @override
  void dispose() {
    _videoController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: background,
      body: Stack(
        children: [
          _buildBackground(),

          Positioned.fill(
            child: Container(
              color: Colors.black.withOpacity(.46),
            ),
          ),

          SafeArea(
            child: Column(
              children: [
                _buildTopBar(),

                Expanded(
                  child: SingleChildScrollView(
                    physics: const BouncingScrollPhysics(),
                    padding: const EdgeInsets.fromLTRB(
                      18,
                      10,
                      18,
                      30,
                    ),
                    child: Center(
                      child: ConstrainedBox(
                        constraints: const BoxConstraints(
                          maxWidth: 600,
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            _buildFeeSummary(),

                            const SizedBox(height: 20),

                            _sectionTitle(
                              'Current Fee',
                              'Latest hostel fee details',
                            ),

                            const SizedBox(height: 12),

                            _buildCurrentFeeCard(),

                            const SizedBox(height: 20),

                            _sectionTitle(
                              'Payment History',
                              'Recent payments for your child',
                            ),

                            const SizedBox(height: 12),

                            _paymentHistoryCard(),
                          ],
                        ),
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

  Widget _buildBackground() {
    if (!_videoController.value.isInitialized) {
      return const Positioned.fill(
        child: ColoredBox(
          color: background,
        ),
      );
    }

    return Positioned.fill(
      child: FittedBox(
        fit: BoxFit.cover,
        child: SizedBox(
          width: _videoController.value.size.width,
          height: _videoController.value.size.height,
          child: VideoPlayer(_videoController),
        ),
      ),
    );
  }

  Widget _buildTopBar() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        10,
        8,
        18,
        8,
      ),
      child: Row(
        children: [
          IconButton(
            onPressed: () => Navigator.pop(context),
            icon: const Icon(
              Icons.arrow_back_ios_new_rounded,
              color: Colors.white,
              size: 20,
            ),
          ),
          const SizedBox(width: 4),
          Expanded(
            child: Text(
              'Fees',
              overflow: TextOverflow.ellipsis,
              style: GoogleFonts.dmSans(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
          Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: Colors.white.withOpacity(.07),
              border: Border.all(
                color: mint.withOpacity(.14),
              ),
            ),
            child: const Icon(
              Icons.account_balance_wallet_outlined,
              color: mint,
              size: 21,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFeeSummary() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            const Color(0xFF00645E).withOpacity(.96),
            const Color(0xFF003B3B).withOpacity(.96),
          ],
        ),
        borderRadius: BorderRadius.circular(28),
        border: Border.all(
          color: mint.withOpacity(.20),
        ),
        boxShadow: [
          BoxShadow(
            color: teal.withOpacity(.13),
            blurRadius: 25,
            spreadRadius: 1,
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Ali Raza',
            style: GoogleFonts.dmSans(
              color: Colors.white,
              fontSize: 21,
              fontWeight: FontWeight.w800,
            ),
          ),

          const SizedBox(height: 3),

          Text(
            'StayBuddy Boys Hostel',
            style: GoogleFonts.dmSans(
              color: Colors.white60,
              fontSize: 11.5,
            ),
          ),

          const SizedBox(height: 18),

          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: [
              _summaryBox(
                title: 'Monthly Fee',
                value: 'Rs. 18,000',
                icon: Icons.payments_outlined,
              ),
              _summaryBox(
                title: 'Paid',
                value: 'Rs. 18,000',
                icon: Icons.check_circle_outline,
                valueColor: green,
              ),
              _summaryBox(
                title: 'Outstanding',
                value: 'Rs. 0',
                icon: Icons.pending_actions_outlined,
                valueColor: green,
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _summaryBox({
    required String title,
    required String value,
    required IconData icon,
    Color valueColor = Colors.white,
  }) {
    return Container(
      width: 160,
      padding: const EdgeInsets.all(13),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(.12),
        borderRadius: BorderRadius.circular(17),
        border: Border.all(
          color: Colors.white.withOpacity(.06),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(
            icon,
            color: mint,
            size: 20,
          ),
          const SizedBox(height: 9),
          Text(
            title,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: GoogleFonts.dmSans(
              color: Colors.white.withOpacity(0.45),
              fontSize: 10.5,
            ),
          ),
          const SizedBox(height: 3),
          Text(
            value,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: GoogleFonts.dmSans(
              color: valueColor,
              fontSize: 15,
              fontWeight: FontWeight.w800,
            ),
          ),
        ],
      ),
    );
  }

  Widget _sectionTitle(
    String title,
    String subtitle,
  ) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: GoogleFonts.dmSans(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.w800,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          subtitle,
          style: GoogleFonts.dmSans(
            color: Colors.white38,
            fontSize: 11,
          ),
        ),
      ],
    );
  }

  Widget _buildCurrentFeeCard() {
    return _glassCard(
      child: Column(
        children: [
          const _FeeDetailRow(
            icon: Icons.calendar_month_outlined,
            label: 'Month',
            value: 'August 2026',
          ),

          const _FeeDetailRow(
            icon: Icons.payments_outlined,
            label: 'Hostel Fee',
            value: 'Rs. 15,000',
          ),

          const _FeeDetailRow(
            icon: Icons.restaurant_outlined,
            label: 'Mess Charges',
            value: 'Rs. 3,000',
          ),

          const _FeeDetailRow(
            icon: Icons.calculate_outlined,
            label: 'Total',
            value: 'Rs. 18,000',
          ),

          const _FeeDetailRow(
            icon: Icons.event_available_outlined,
            label: 'Due Date',
            value: '31 Aug 2026',
          ),

          const SizedBox(height: 14),

          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(
              horizontal: 14,
              vertical: 12,
            ),
            decoration: BoxDecoration(
              color: green.withOpacity(.10),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(
                color: green.withOpacity(.22),
              ),
            ),
            child: Row(
              children: [
                const Icon(
                  Icons.verified_rounded,
                  color: green,
                  size: 20,
                ),
                const SizedBox(width: 9),
                Expanded(
                  child: Text(
                    'Fee for August has been paid successfully.',
                    style: GoogleFonts.dmSans(
                      color: Colors.white,
                      fontSize: 11.5,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
                Text(
                  'PAID',
                  style: GoogleFonts.dmSans(
                    color: green,
                    fontSize: 11,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _paymentHistoryCard() {
    return _glassCard(
      child: Column(
        children: [
          _paymentItem(
            month: 'August 2026',
            amount: 'Rs. 18,000',
            date: 'Paid on 24 Aug 2026',
            status: 'Paid',
            statusColor: green,
          ),

          _divider(),

          _paymentItem(
            month: 'July 2026',
            amount: 'Rs. 18,000',
            date: 'Paid on 26 Jul 2026',
            status: 'Paid',
            statusColor: green,
          ),

          _divider(),

          _paymentItem(
            month: 'June 2026',
            amount: 'Rs. 18,000',
            date: 'Paid on 25 Jun 2026',
            status: 'Paid',
            statusColor: green,
          ),

          _divider(),

          _paymentItem(
            month: 'May 2026',
            amount: 'Rs. 18,000',
            date: 'Paid on 30 May 2026',
            status: 'Late',
            statusColor: warning,
          ),
        ],
      ),
    );
  }

  Widget _paymentItem({
    required String month,
    required String amount,
    required String date,
    required String status,
    required Color statusColor,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(
        vertical: 13,
      ),
      child: Row(
        children: [
          Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(
              color: statusColor.withOpacity(.10),
              borderRadius: BorderRadius.circular(13),
            ),
            child: Icon(
              Icons.receipt_long_rounded,
              color: statusColor,
              size: 21,
            ),
          ),

          const SizedBox(width: 12),

          Expanded(
            child: Column(
              crossAxisAlignment:
                  CrossAxisAlignment.start,
              children: [
                Text(
                  month,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontSize: 12.5,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  date,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: Colors.white38,
                    fontSize: 10.5,
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(width: 8),

          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(
                amount,
                style: GoogleFonts.dmSans(
                  color: Colors.white,
                  fontSize: 11.5,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 3),
              Text(
                status,
                style: GoogleFonts.dmSans(
                  color: statusColor,
                  fontSize: 10.5,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _divider() {
    return Divider(
      height: 1,
      color: Colors.white.withOpacity(.06),
    );
  }

  Widget _glassCard({
    required Widget child,
  }) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(.32),
        borderRadius: BorderRadius.circular(22),
        border: Border.all(
          color: mint.withOpacity(.12),
        ),
      ),
      child: child,
    );
  }
}

class _FeeDetailRow extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;

  const _FeeDetailRow({
    required this.icon,
    required this.label,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    const mint = Color(0xFF85FFF2);

    return Container(
      padding: const EdgeInsets.symmetric(
        vertical: 13,
      ),
      decoration: BoxDecoration(
        border: Border(
          bottom: BorderSide(
            color: Colors.white.withOpacity(.06),
          ),
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(
              color: mint.withOpacity(.07),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(
              icon,
              color: mint,
              size: 19,
            ),
          ),

          const SizedBox(width: 11),

          Expanded(
            child: Text(
              label,
              style: GoogleFonts.dmSans(
                color: Colors.white54,
                fontSize: 11.5,
              ),
            ),
          ),

          const SizedBox(width: 10),

          Flexible(
            child: Text(
              value,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              textAlign: TextAlign.end,
              style: GoogleFonts.dmSans(
                color: Colors.white,
                fontSize: 11.5,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
